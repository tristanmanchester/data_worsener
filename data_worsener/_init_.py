# data_worsener/__init__.py

# Import main functions from artifacts
from .artifacts.cupping_artifacts import add_edge_cupping_artefact
from .artifacts.ring_artifacts import create_rings, create_large_ring

# Import main functions from preprocessing
from .preprocessing.segmentation import reduce_pore_contrast
from .preprocessing.sinogram import remove_stripe_based_sorting

# Import main functions from segmentation
from .segmentation.segment_h5 import process_h5_files as segment_volume
from .segmentation.segment_h5 import resize_slice, get_thresholds, apply_thresholds

# Import utility functions
from .utils.io import process_h5_files as worsen_volume
from .utils.visualization import save_test_outputs

# Version info
__version__ = '0.1.0'

# Define what gets imported with "from data_worsener import *"
__all__ = [
    # Artifacts
    'add_edge_cupping_artefact',
    'create_rings',
    'create_large_ring',

    # Preprocessing
    'reduce_pore_contrast',
    'remove_stripe_based_sorting',

    # Segmentation
    'segment_volume',
    'resize_slice',
    'get_thresholds',
    'apply_thresholds',

    # Utils
    'worsen_volume',
    'save_test_outputs',
]