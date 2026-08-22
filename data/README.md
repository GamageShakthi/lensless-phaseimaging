# Data Directory

This directory stores PSF data and test images.

## Getting Data

### Option 1: Generate Synthetic Data (No Downloads!)
The code can generate all PSFs and test scenes synthetically:
```bash
python scripts/simulate_measurement.py --psf_type random_phase_mask
```

### Option 2: Real DiffuserCam PSFs
Download measured PSFs from the Waller Lab:
- [DiffuserCam Tutorial Data](https://waller-lab.github.io/DiffuserCam/tutorial)
- Place `psf.tiff` in this directory

### Option 3: Natural Images for Training
Use any image dataset:
- [BSD500](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html)
- [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
- Or just use your own photos!

Place images in `data/train/` and `data/val/` subdirectories.

## File Formats
- `.npy` — NumPy arrays (for PSFs and intermediate data)
- `.png` / `.jpg` — Standard images
- `.tiff` — High dynamic range images (16-bit)
