# Cell volume tracking Toolkit

Segment, track, and quantify cell volume over time. The toolkit provides:

* `segmentation.py` — segments cells in time-lapse TIFF stacks, tiles large fields of view, tracks instances across frames, compute integrated volume, and saves per-tile videos and label stacks.  

---

## Quick start

### Option A — Docker (recommended)

Download existing image (recommended):

```bash
docker pull mdyakova/cell_tracking:v1
```

or build the image (pre-caches Cellpose CPSAM weights during build for faster first run):

```bash
docker build -t cell_tracking:v1 .
```

Run segmentation + tracking (with GPU and a bind mount; adjust paths as needed):

```bash
docker run --rm --gpus all -v "C:\work_dir\cell_tracking_files:/cell_tracking_files" mdyakova/cell_tracking:v1 python segmentation.py --image_directory "/cell_tracking_files/data" --output_directory "/cell_tracking_files/tracking_results" --name_filter Xenopus --cell_diameter_min 30 --cell_diameter_max 100 --tile_size 400
```

### Option B — Local Python (no Docker)

1. Install Python 3.8 and dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

2. Run:

```bash
python3.8 main/segmentation.py \
  --image_directory "../Avik/20250624_All_Images_for_Masha/New Turgor Analysis" \
  --output_directory "../Avik/tracking_092225" \
  --tile_size 400
  --name_filter axl
  --cell_diameter_min 30 
  --cell_diameter_max 100

```

Dependencies are pinned in `requirements.txt`. 

---

## What the scripts do

### `segmentation.py` (segmentation + tracking)

* Loads a Cellpose **omnipose_cyto** model on GPU and processes each frame.
* Splits each 3D stack into overlapping XY tiles or whole image; runs segmentation for the cell size range between the minimal and maximum values; merges masks that correspond to the same cell; tracks them across frames; compute integrated volumes for each cell; and saves per-tile outputs:

  * `*_x_{j}_y_{i}.tif` (uint16 intensity video)
  * `*_x_{j}_y_{i}_labels.tif` (uint16 label video)
  * `processed_files.txt` (append-only log to skip completed tiles)
    Parameters include `--cell_diameter_min` and `--cell_diameter_max` for cell range and `--name_filter` to restrict which files run.  

* Outputs:
  * `<output>/results<file>.csv` — integrated volumes for each segmented cell

---

## Input/Output layout (typical)

```
image_directory/
  sampleA_... .tif        # time-lapse z=frames, y, x
output_directory/
  sampleA_x_000_y_000.tif # if tile size less than image size
  sampleA_x_000_y_000_labels.tif
  processed_files.txt
  results_sampleA_x_000_y_000.csv

```


## Command-line arguments

### `segmentation.py`

* `--image_directory` (str, required): folder with TIFF stacks (searched recursively).
* `--output_directory` (str, required): where outputs are written.
* `--tile_size` (int, required): tile edge in pixels (overlap of 50 px is applied).
* `--name_filter` (str, default `""`): run only files whose path contains this substring.
* `--cell_diameter_min` (int, optional): minimum cell size
* `--cell_diameter_max` (int, optional): maximum cell size

---

## Requirements

* Python 3.8
* CUDA-capable GPU (recommended) + NVIDIA drivers for the Docker `--gpus all` option
* Packages pinned in `requirements.txt` (Cellpose 4.0.4, trackpy 0.7, scikit-image 0.21, OpenCV 4.11, etc.). 

The provided Dockerfile uses `python:3.8-slim`, adds GUI libs for OpenCV wheels, installs requirements, copies `main/` as the working directory, **and pre-caches CPSAM weights** during the image build so first inference runs faster. 

---

## Tips & troubleshooting

* **GPU vs CPU:** The scripts request GPU for Cellpose (`CellposeModel(gpu=True)`). If you must run on CPU, change to `gpu=False` in code or ensure no GPU is visible in the container. 
* **Large frames:** Use a `--tile_size` that fits GPU memory. Tiles overlap by 50 px internally; outputs are per tile. 
* **Skipping finished tiles:** The pipeline appends file names to `processed_files.txt`. Delete entries to re-process tiles. 

---

## Citation

If you use this toolkit in your research, please cite Cellpose and trackpy as appropriate and reference this repository.

---

## License

MIT License

---

## Acknowledgements

* Segmentation powered by **Cellpose**.
* Merging of  predictions uses custom contour-merging utilities. 

---

**Repository layout**

```
main/
  segmentation.py
  utilities.py
  requirements.txt
  Dockerfile
  README.md  <-- (this file)
```

* `segmentation.py` — per-cell feature extraction & visualization. 
* `utilities.py` — label-merging helpers. 
* `requirements.txt` — pinned deps for reproducibility. 

---

### 📫 **Contact**
For questions or contributions, please contact:
**Mariia Diakova**
- GitHub: [MDyakova](https://github.com/MDyakova)
- email: m.dyakova.ml@gmail.com
