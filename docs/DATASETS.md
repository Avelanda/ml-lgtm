# Dataset Preparation

## DL3DV

We use the [DL3DV-10K](https://github.com/DL3DV-10K/Dataset) dataset. For high-resolution training, we use the **4K** version (3840x2160). Lower resolutions are also supported.

`DL3DV/DL3DV-Benchmark` contains ~140 test scenes, however, ~130 of them are also inside `DL3DV/DL3DV-ALL-4K` (and other resolutions). This code will exclude these scenes from the train set. The test set is built from the index json file with the `DL3DV/DL3DV-ALL-4K` dataset. In DL3DV's terminology, "4K" and "2K" refers to the horizontal resolution being approximately 4,000 and 2,000 pixels, respectively. Their "960P" (540x960) is actually 540P and their "480P" (270x480) is actually 270P, where typically "P" refers to the vertical resolution.

### Step 1: Download

The DL3DV dataset is hosted on HuggingFace. You need to:

1. Create a HuggingFace account and request access to the DL3DV dataset repos (e.g., [DL3DV-ALL-4K](https://huggingface.co/datasets/DL3DV/DL3DV-ALL-4K)).
2. Set your HuggingFace token:
   ```bash
   export HF_TOKEN=hf_xxxxxxxxxxxxx
   # or
   hf auth login
   ```

Download data one subset at a time. The dataset is organized into subsets: 1K, 2K, ..., 11K.

```bash
# Download 4K resolution, subset "1K" (scenes 0-1000)
python scripts/download_dl3dv.py \
    --output_dir data/dl3dv-4k \
    --resolution 4K \
    --subset 1K \
    --clean_cache

# Download all subsets (run for each subset)
for subset in 1K 2K 3K 4K 5K 6K 7K 8K 9K 10K 11K; do
    python scripts/download_dl3dv.py \
        --output_dir data/dl3dv-4k \
        --resolution 4K \
        --subset $subset \
        --clean_cache
done
```

### Step 2: Preprocess

Convert the downloaded data into `.torch` chunk files:

```bash
python scripts/preprocess_dl3dv.py \
    --input_dir data/dl3dv-4k \
    --output_dir data/dl3dv \
    --resolution 4K
```

### DL3DV Data Format

Each `.torch` chunk is a list of scene dictionaries:

```python
{
    "key": "1K/scene_hash",          # Scene identifier
    "timestamps": Tensor[int64],     # Frame indices
    "cameras": Tensor[float32],      # (N, 18) normalized intrinsics + extrinsics
    "images": [Tensor[uint8], ...],  # List of JPEG-encoded byte tensors
}
```

The 18-dimensional camera vector:

- `[0:6]`: Normalized intrinsics `(fx/w, fy/h, cx/w, cy/h, 0, 0)`
- `[6:18]`: Flattened 3x4 world-to-camera matrix (OpenCV convention)
