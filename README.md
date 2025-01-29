# Data Worsener

This project provides tools for transforming high-quality laboratory CT data to simulate synchrotron imaging characteristics, creating training data for deep learning segmentation models. The methodology, demonstrated through copper oxide dissolution studies, achieves over 80% segmentation accuracy on unseen data while reducing processing time from hours to 30 seconds per volume.

## Overview

The project consists of two main components:

1. **Data Worsening Pipeline**: Transforms high-quality laboratory CT data to mimic synchrotron imaging characteristics through:
   - Initial processing (cropping, masking)
   - Artifact introduction (cupping, rings)
   - Sinogram processing
   - Final reconstruction

2. **Segmentation Pipeline**: Processes the data using multi-otsu thresholding to create binary segmentations.

<p align="center">
  <img src="https://github.com/tristanmanchester/data_worsener/blob/main/full_process.png" width="700">
</p>

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/data_worsener.git
cd data_worsener

# Install the package in development mode
pip install -e .
```

## Usage

The package provides two main scripts:

### 1. Data Worsening

Transform high-quality CT data to simulate synchrotron characteristics:

```bash
python scripts/worsen_data.py -i /path/to/input -o /path/to/output [--test-only] [--skip-test] [--force]
```

Options:
- `--test-only`: Only process middle slice of first file
- `--skip-test`: Skip test and process all files
- `--force`: Process without confirmation

### 2. Segmentation

Segment the CT data using multi-otsu thresholding:

```bash
python scripts/segment.py -i /path/to/input -o /path/to/output [-c CHUNKS] [--test-only]
```

Options:
- `-c CHUNKS`: Number of chunks to process (default: 8)
- `--test-only`: Only process middle slice of first file

## Features

- Automated preprocessing pipeline for CT data
- Simulation of common synchrotron artifacts:
  - Ring artifacts
  - Edge enhancement effects
  - Intensity variations
- Efficient data processing with GPU acceleration
- Support for large datasets through chunked processing
- Configurable processing parameters

## System Requirements

- NVIDIA GPU with CUDA support (tested on RTX 4070)
- 8GB+ GPU memory
- Python 3.7+
- Required packages:
  - cupy
  - numpy
  - opencv-python
  - scikit-image
  - h5py
  - tqdm
  - matplotlib

## Citation

If you use this code in your research, please cite:

```bibtex
Me
```

## Acknowledgements
