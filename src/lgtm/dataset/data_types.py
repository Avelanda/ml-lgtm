"""
Typed view/batch structures and camera conventions for LGTM.

For licensing see accompanying LICENSE file.
Copyright (C) 2026 Apple Inc. All Rights Reserved.

For third-party code see ACKNOWLEDGMENTS file.

Camera convention used throughout the codebase:

- poses:      c2w 4x4 matrices. The translation column [:3, 3] gives
              the camera position in world coordinates.
- intrinsics: Normalized intrinsic 3x3 matrices (first row divided by
              image width, second row divided by image height).
- Ks:         Pixel-scale intrinsic 3x3 matrices (denormalized).
- Ts / w2c:   w2c 4x4 matrices. Ts = poses.inverse().
"""

from typing import Callable, Literal, TypedDict

from jaxtyping import Float, Int64
from torch import Tensor

DataStage = Literal["train", "test"]
Mode = Literal["train", "test"]


# The following types mainly exist to make type-hinted keys show up in VS Code.
# Some dimensions are annotated as "_" because either:
# 1. They're expected to change as part of a function call.
#    (e.g., resizing the dataset).
# 2. They're expected to vary within the same function call.
# (e.g., the number of views, which differs between context and.
# Target BatchedViews).


class BatchedViews(TypedDict, total=False):
    poses: Float[Tensor, "batch _ 4 4"]  # c2w 4x4
    intrinsics: Float[Tensor, "batch _ 3 3"]  # normalized intrinsics 3x3
    image: Float[Tensor, "batch _ _ _ _"]  # batch view channel height width
    near: Float[Tensor, "batch _"]  # batch view
    far: Float[Tensor, "batch _"]  # batch view
    index: Int64[Tensor, "batch _"]  # batch view
    overlap: Float[Tensor, "batch _"]  # batch view

    def __repr__(self):
        tensor_repr = lambda t: (
            f"{t.dtype} {tuple(t.shape)}" if t is not None else "None"
        )
        poses_repr = tensor_repr(self.get("poses"))
        intrinsics_repr = tensor_repr(self.get("intrinsics"))
        image_repr = tensor_repr(self.get("image"))
        near_repr = tensor_repr(self.get("near"))
        far_repr = tensor_repr(self.get("far"))
        index_repr = tensor_repr(self.get("index"))
        overlap_repr = tensor_repr(self.get("overlap"))

        rc = "BatchedViews(\n"
        rc += f"  poses     : {poses_repr},\n"
        rc += f"  intrinsics: {intrinsics_repr},\n"
        rc += f"  image     : {image_repr},\n"
        rc += f"  near      : {near_repr},\n"
        rc += f"  far       : {far_repr},\n"
        rc += f"  index     : {index_repr},\n"
        rc += f"  overlap   : {overlap_repr},\n"
        rc += ")"
        return rc

    def __str__(self):
        return self.__repr__()


class BatchedExample(TypedDict, total=False):
    target: BatchedViews
    context: BatchedViews
    scene: list[str]

    def __repr__(self):
        target_repr = "\n".join(
            f"    {line}" for line in self["target"].__repr__().split("\n")
        )

        if "context" in self:
            context_repr = "\n".join(
                f"    {line}" for line in self["context"].__repr__().split("\n")
            )
        else:
            context_repr = "    None"

        if "scene" in self:
            scene_repr = str(self["scene"])
        else:
            scene_repr = "None"

        rc = "BatchedExample(\n"
        rc += f"  target : {target_repr},\n"
        rc += f"  context: {context_repr},\n"
        rc += f"  scene  : {scene_repr},\n"
        rc += ")"
        return rc

    def __str__(self):
        return self.__repr__()


class UnbatchedViews(TypedDict, total=False):
    poses: Float[Tensor, "_ 4 4"]  # c2w 4x4
    intrinsics: Float[Tensor, "_ 3 3"]  # normalized intrinsics 3x3
    image: Float[Tensor, "_ 3 height width"]
    near: Float[Tensor, " _"]
    far: Float[Tensor, " _"]
    index: Int64[Tensor, " _"]


class UnbatchedExample(TypedDict, total=False):
    target: UnbatchedViews
    context: UnbatchedViews
    scene: str


DataShim = Callable[[BatchedExample], BatchedExample]

AnyExample = BatchedExample | UnbatchedExample
AnyViews = BatchedViews | UnbatchedViews
