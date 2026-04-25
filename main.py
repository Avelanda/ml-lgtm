"""
Training and evaluation entry point for LGTM.

For licensing see accompanying LICENSE file.
Copyright (C) 2026 Apple Inc. All Rights Reserved.

For third-party code see ACKNOWLEDGMENTS file.
"""

import PIL.Image

PIL.Image.MAX_IMAGE_PIXELS = 1_000_000_000

from pathlib import Path

import hydra
import rootutils
import torch
import wandb
from colorama import Fore
from hydra.core.hydra_config import HydraConfig
from jaxtyping import install_import_hook
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers.wandb import WandbLogger
from omegaconf import DictConfig, OmegaConf

import lgtm.storage as storage
from lgtm.utils.checkpoint_utils import (
    ModelCheckpointWithSync,
    resolve_test_checkpoint_path,
)
from lgtm.utils.logging_utils import (
    LocalLogger,
    get_exp_output_dir,
    get_wandb_api_key,
    load_wandb_run_id,
    save_wandb_run_id,
)
from lgtm.utils.model_utils import (
    checkpoint_filter_fn,
    filter_state_dict_by_prefix,
)
from lgtm.utils.training_utils import StepTracker

with install_import_hook(
    ("lgtm",),
    ("beartype", "beartype"),
):
    from lgtm.config import load_typed_root_config, set_cfg
    from lgtm.dataset.data_module import DataModule
    from lgtm.engine import LGTMEngine
    from lgtm.loss import get_losses
    from lgtm.model.encoder import get_encoder
    from lgtm.model.gaussian.gaussian_rasterizer import get_rasterizer


# Set up infrastructure environment variables when available.
storage.setup_environment()

_project_root = rootutils.find_root(
    search_from=Path(__file__),
    indicator="pyproject.toml",
)


def exp_name_resolver() -> str:
    """
    Determines the exp_name based on the "experiment" config group,
    which corresponds to the path (stem) of the config file.

    Example:
        command  : python main.py +experiment=dl3dv_abc  (user's command)
        config   : config/experiment/dl3dv_abc.yaml  (hydra loads this)
        exp_name : dl3dv_abc  (resolved by this function)
        outputdir: outputs/dl3dv_abc  (hydra sets ["runtime"]["output_dir"])

    Raises:
        ValueError: If the experiment config cannot be determined or
            contains .yaml extension.
    """
    try:
        choices = HydraConfig.get().runtime.choices
        exp_name = choices.get("experiment")
        if exp_name is None:
            raise ValueError(
                "Hydra config group 'experiment' must be selected (e.g., using"
                " +experiment=...)"
            )
        exp_name = str(exp_name)  # Ensure it's a string

        # Check for .yaml extension and reject it.
        if exp_name.endswith(".yaml"):
            raise ValueError(
                f"Experiment name '{exp_name}' should not include '.yaml'"
                f" extension. Use '{exp_name[:-5]}' instead."
            )

        return exp_name
    except Exception as e:
        print(f"Error in exp_name_resolver: {e}")
        raise ValueError(
            "Could not determine experiment name during resolver execution."
        ) from e


OmegaConf.register_new_resolver("exp_name_resolver", exp_name_resolver)


def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"


@hydra.main(
    version_base=None,
    config_path=str(_project_root / "configs"),
    config_name="main",
)
def main(cfg_dict: DictConfig):
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)

    # Determine output_dir and exp_name from hydra.
    output_dir = Path(
        hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
    )
    exp_name = HydraConfig.get().runtime.choices.get("experiment")

    # Apply the same validation as exp_name_resolver.
    if exp_name and exp_name.endswith(".yaml"):
        raise ValueError(
            f"Experiment name '{exp_name}' should not include '.yaml'"
            f" extension. Use '{exp_name[:-5]}' instead."
        )
    wandb_name_prev = cfg_dict.wandb.name
    if wandb_name_prev == "automatic":
        cfg_dict.wandb.name = f"{exp_name}"

    # Add cluster task ID to wandb name for better tracking.
    task_id = storage.get_task_id()
    if task_id:
        cfg_dict.wandb.name = f"{cfg_dict.wandb.name}_{task_id}"
        print(cyan(f"task_id: {task_id}"))
    print(cyan(f"output_dir: {output_dir}"))
    print(cyan(f"exp_name  : {exp_name}"))
    print(cyan(f"wandb_name: {cfg_dict.wandb.name} (prev: {wandb_name_prev})"))

    # Set up logging with wandb.
    callbacks = []
    if cfg_dict.wandb.mode != "disabled":
        # Login (rank 0 only?)
        wandb_api_key = get_wandb_api_key()
        if wandb_api_key is not None:
            wandb.login(key=wandb_api_key)

        # Handle wandb run ID persistence for training/testing continuity.
        wandb_run_id = None
        if cfg.mode == "test":
            # For testing, try to load the existing run ID from training.
            wandb_run_id = load_wandb_run_id()
            if wandb_run_id:
                print(f"Resuming Wandb run with ID: {wandb_run_id}")

        logger = WandbLogger(
            project=cfg_dict.wandb.project,
            mode=cfg_dict.wandb.mode,
            name=cfg_dict.wandb.name,
            tags=cfg_dict.wandb.get("tags", None),
            log_model=False,  # Do not upload model to wandb
            save_dir=output_dir,
            config=OmegaConf.to_container(cfg_dict),
            id=wandb_run_id,
            resume="must" if wandb_run_id else None,
        )
        callbacks.append(LearningRateMonitor("step", True))

        # On rank != 0, wandb.run is None.
        if wandb.run is not None:
            # https://docs.wandb.ai/guides/integrations/add-wandb-to-any-library/#prevent-x-axis-misalignments
            # Define the default x-axis metric for all logged values.
            wandb.define_metric("*", step_metric="global_step")
            wandb.run.log_code("src/lgtm")
    else:
        logger = LocalLogger()

    # Set up checkpointing with optional remote sync.
    if cfg.checkpointing.sync_to_remote_after_ckpt_save:
        checkpoint_callback = ModelCheckpointWithSync(
            dirpath=output_dir / "checkpoints",
            output_dir=output_dir,
            every_n_train_steps=cfg.checkpointing.every_n_train_steps,
            save_top_k=cfg.checkpointing.save_top_k,
            save_weights_only=cfg.checkpointing.save_weights_only,
            monitor="info/global_step",
            mode="max",
            sync_enabled=True,
            sync_remove_deleted=True,
            sync_quiet=False,
        )
        print(cyan("Checkpoint sync to remote enabled"))
    else:
        # Use regular ModelCheckpoint without sync.
        checkpoint_callback = ModelCheckpoint(
            output_dir / "checkpoints",
            every_n_train_steps=cfg.checkpointing.every_n_train_steps,
            save_top_k=cfg.checkpointing.save_top_k,
            save_weights_only=cfg.checkpointing.save_weights_only,
            monitor="info/global_step",
            mode="max",
        )

    callbacks.append(checkpoint_callback)
    callbacks[-1].CHECKPOINT_EQUALS_CHAR = "_"

    # Handle 0-step training case early - skip training entirely.
    # Before any setup.
    if cfg.mode == "train" and cfg.trainer.max_steps == 0:
        print(cyan("Skipping training due to max_steps=0 (no-op training)"))
        return

    # This allows the current step to be shared with the data loader processes.
    step_tracker = StepTracker()

    devices = "auto"
    strategy = (
        "ddp_find_unused_parameters_true"
        if torch.cuda.device_count() > 1
        else "auto"
    )
    num_nodes = cfg.trainer.num_nodes
    print(
        "Lightning Trainer: "
        f"devices={devices}, strategy={strategy}, num_nodes={num_nodes}"
    )

    trainer = Trainer(
        max_epochs=-1,
        num_nodes=num_nodes,
        accelerator="gpu",
        logger=logger,
        devices=devices,
        strategy=strategy,
        callbacks=callbacks,
        val_check_interval=cfg.trainer.val_check_interval,
        check_val_every_n_epoch=None,
        enable_progress_bar=False,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        max_steps=cfg.trainer.max_steps,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        inference_mode=(
            False if (cfg.mode == "test" and cfg.test.enable_pose_opt) else True
        ),
    )
    torch.manual_seed(cfg_dict.seed + trainer.global_rank)
    encoder = get_encoder(cfg.encoder)

    # Parse test_ckpt_path: used in "test" mode.
    if cfg.mode == "test" and cfg.checkpointing.test_ckpt_path:
        test_ckpt_path = cfg.checkpointing.test_ckpt_path

        # Resolve checkpoint path with fallback search strategy.
        test_ckpt_path = resolve_test_checkpoint_path(
            test_ckpt_path=test_ckpt_path,
            exp_output_dir=get_exp_output_dir(),
            maybe_use_remote_ckpt=cfg.checkpointing.maybe_use_remote_ckpt,
        )
    else:
        test_ckpt_path = None

    # Load the pretrained encoder weights: used in "train" mode.
    if cfg.mode == "train" and cfg.encoder.pretrained_checkpoint_path:
        pretrained_ckpt_path = cfg.encoder.pretrained_checkpoint_path

        print(f"Loading pretrained weights from {pretrained_ckpt_path}")
        pretrained_ckpt_path = storage.resolve_path(pretrained_ckpt_path)

        ckpt_weights = torch.load(
            pretrained_ckpt_path, map_location="cpu", weights_only=False
        )
        if "flash3d" in str(pretrained_ckpt_path):
            encoder.gaussian_predictor.load_model(pretrained_ckpt_path)
        elif "model" in ckpt_weights:
            # Typically for weights from MASt3R, e.g. MASt3R_ViTLarge_xxx.pth.
            print('Loading weights from "model" key')
            ckpt_weights = ckpt_weights["model"]
            ckpt_weights = checkpoint_filter_fn(ckpt_weights, encoder)
            missing_keys, unexpected_keys = encoder.load_state_dict(
                ckpt_weights, strict=False
            )
            print(f"- len(missing_keys): {len(missing_keys)}")
            print(f"- len(unexpected_keys): {len(unexpected_keys)}")
        elif "state_dict" in ckpt_weights:
            # Typically for weights from noposplat, e.g. re10.ckpt.
            print('Loading weights from "state_dict" key')
            ckpt_weights = ckpt_weights["state_dict"]

            # Strip "encoder." prefix, as the encoder obj is used.
            # To load the weights.
            ckpt_weights = {
                k[8:]: v
                for k, v in ckpt_weights.items()
                if k.startswith("encoder.")
            }

            # Filter keys based on ignored prefixes.
            ignored_prefixes = getattr(
                cfg.encoder,
                "pretrained_checkpoint_ignored_prefixes",
                None,
            )
            ckpt_weights = filter_state_dict_by_prefix(
                state_dict=ckpt_weights,
                ignored_prefixes=ignored_prefixes,
            )

            # Load.
            missing_keys, unexpected_keys = encoder.load_state_dict(
                ckpt_weights, strict=False
            )
            print(f"- len(missing_keys): {len(missing_keys)}")
            print(f"- len(unexpected_keys): {len(unexpected_keys)}")
        else:
            raise ValueError(
                f"Invalid checkpoint format: {pretrained_ckpt_path}"
            )

    # Load global losses and pose_opt.enabled_losses.
    losses = get_losses(cfg.loss)
    name_to_loss = {loss.name: loss for loss in losses}
    pose_opt_losses = []
    for loss_name in cfg.pose_opt.enabled_losses:
        if loss_name not in name_to_loss:
            available_losses = list(name_to_loss.keys())
            raise ValueError(
                f"pose_opt.enabled_losses contains invalid loss '{loss_name}'. "
                f"Available training losses: {available_losses}"
            )
        pose_opt_losses.append(name_to_loss[loss_name])

    if trainer.global_rank == 0:
        print(f"train losses: {[loss.name for loss in losses]}")
        if cfg.test.enable_pose_opt:
            print(f"pose_opt losses: {[loss.name for loss in pose_opt_losses]}")

    # Propagate shared texture settings from encoder to rasterizer so they.
    # Only need to be specified once (under encoder.gaussian).
    gaussian_cfg = cfg.encoder.gaussian
    rasterizer_cfg = cfg.rasterizer
    rasterizer_cfg.texture_alpha_mode = gaussian_cfg.texture_alpha_mode
    rasterizer_cfg.texture_color_sigma = gaussian_cfg.texture_color_sigma

    engine = LGTMEngine(
        optimizer_cfg=cfg.optimizer,
        train_cfg=cfg.train,
        test_cfg=cfg.test,
        pose_opt_cfg=cfg.pose_opt,
        encoder=encoder,
        rasterizer=get_rasterizer(rasterizer_cfg),
        losses=losses,
        pose_opt_losses=pose_opt_losses,
        step_tracker=step_tracker,
    )
    # Determine target_image_shape from mode-specific config.
    if cfg.mode == "train" and cfg.train.train_target_image_shape:
        target_image_shape = tuple(cfg.train.train_target_image_shape)
    elif cfg.mode == "test" and cfg.test.test_target_image_shape:
        target_image_shape = tuple(cfg.test.test_target_image_shape)
    else:
        target_image_shape = None

    data_module = DataModule(
        dataset_cfgs=cfg.dataset,
        data_loader_cfg=cfg.data_loader,
        mode=cfg.mode,
        step_tracker=step_tracker,
        global_rank=trainer.global_rank,
        target_image_shape=target_image_shape,
    )

    print(cyan(f"Running cfg.mode = {cfg.mode}"))
    if cfg.mode == "train":
        # Save wandb run ID for training mode (rank 0 only) before training.
        if (
            cfg_dict.wandb.mode != "disabled"
            and trainer.global_rank == 0
            and hasattr(logger, "experiment")
            and logger.experiment is not None
        ):
            # Use logger.experiment.id which is reliably available immediately.
            current_wandb_run_id = logger.experiment.id

            # Only save if this is a new run (not resuming)
            existing_wandb_run_id = load_wandb_run_id()
            if existing_wandb_run_id != current_wandb_run_id:
                print(f"Saving new wandb run ID: {current_wandb_run_id}")
                save_wandb_run_id(current_wandb_run_id)
            else:
                print(
                    cyan(
                        "Continuing existing wandb run ID:"
                        f" {current_wandb_run_id}"
                    )
                )

            # Verify the file was created.
            wandb_id_path = get_exp_output_dir() / "wandb_run_id.txt"
            if not wandb_id_path.exists():
                raise RuntimeError(
                    f"Wandb run ID file should exist at {wandb_id_path}"
                )
            print(f"Wandb run ID saved to: {wandb_id_path}")

            # Set cluster status message with wandb URL for easy access.
            wandb_entity = getattr(cfg_dict.wandb, "entity", "")
            wandb_url = f"https://wandb.ai/{wandb_entity}/{cfg_dict.wandb.project}/runs/{current_wandb_run_id}"
            storage.set_status_message(wandb_url)
            if storage.is_available():
                print(cyan(f"Set status message: {wandb_url}"))

        trainer.fit(engine, datamodule=data_module)

        # Synchronize output folder to remote storage.
        if trainer.global_rank == 0 and cfg.sync_to_remote_on_finish:
            print("[sync] Syncing output folder after training...")
            storage.sync_output_dir(
                source_dir=output_dir,
                update=True,
                remove_deleted=True,
                quiet=True,
            )
            print("[sync] Sync completed after training.")
    elif cfg.mode == "test":
        if test_ckpt_path is None:
            raise ValueError("test_ckpt_path must be specified for test mode")

        # Clean up test_dataloader.log.
        if trainer.global_rank == 0:
            test_dataloader_log_path = (
                get_exp_output_dir() / "test_dataloader.log"
            )
            if test_dataloader_log_path.exists():
                test_dataloader_log_path.unlink()
                print(
                    "Cleaned up test_dataloader.log:"
                    f" {test_dataloader_log_path}"
                )

        # Run multi-GPU test.
        trainer.test(
            engine,
            datamodule=data_module,
            ckpt_path=test_ckpt_path,
        )

        # Synchronize output folder to remote storage.
        if trainer.global_rank == 0 and cfg.sync_to_remote_on_finish:
            print("[sync] Syncing output folder after testing...")
            storage.sync_output_dir(
                source_dir=output_dir,
                update=True,
                remove_deleted=True,
                quiet=True,
            )
            print("[sync] Sync completed after testing.")
    else:
        raise ValueError(
            f"Unknown mode: {cfg.mode}. Expected 'train' or 'test'"
        )


if __name__ == "__main__":
    main()
