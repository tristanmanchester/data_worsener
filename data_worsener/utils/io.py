import os
import h5py
import numpy as np
import cupy as cp
from tqdm import tqdm
from skimage.transform import radon, iradon

from ..preprocessing import reduce_pore_contrast, remove_stripe_based_sorting
from ..artifacts import add_edge_cupping_artefact, create_rings, create_large_ring
from .visualization import setup_visualization, update_display, save_test_outputs


def process_h5_files(input_folder, output_folder):
    """
    Process all H5 files in the input folder and save results to output folder.

    Parameters:
    -----------
    input_folder : str
        Path to folder containing input H5 files
    output_folder : str
        Path to folder where processed files will be saved
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f'Created output directory: {output_folder}')

    # Setup visualization
    fig, (ax1, ax2), img1, img2 = setup_visualization()

    h5_files = [f for f in os.listdir(input_folder) if f.endswith('.h5')]
    print(f'Found {len(h5_files)} H5 files to process')

    for file_idx, h5_file in enumerate(h5_files, 1):
        input_path = os.path.join(input_folder, h5_file)
        output_path = os.path.join(output_folder, h5_file)
        print(f'\n[{file_idx}/{len(h5_files)}] Processing file: {h5_file}')

        with h5py.File(input_path, 'r') as f_in:
            data = f_in['data']
            total_slices = data.shape[0]

            with h5py.File(output_path, 'w') as f_out:
                dset_out = f_out.create_dataset('data',
                                                shape=data.shape,
                                                dtype=np.uint16)

                print(f'Processing {total_slices} slices...')
                for slice_idx in tqdm(range(total_slices)):
                    slice_data = data[slice_idx]

                    try:
                        # Update display for original slice
                        update_display(img1, ax1, slice_data,
                                       f'Original Slice {slice_idx}/{total_slices}')

                        # Process slice
                        processed_slice = process_single_slice(slice_data)

                        # Update display for processed slice
                        update_display(img2, ax2, processed_slice,
                                       f'Processed Slice {slice_idx}/{total_slices}')

                        dset_out[slice_idx] = processed_slice

                    except Exception as e:
                        print(f'\nError processing slice {slice_idx} in {h5_file}: {str(e)}')
                        dset_out[slice_idx] = slice_data

                    cp.get_default_memory_pool().free_all_blocks()

        print(f'Completed processing {h5_file}')


def process_single_slice(slice_data):
    """
    Process a single CT slice with all artifacts and reconstruction.

    Parameters:
    -----------
    slice_data : ndarray
        Input CT slice

    Returns:
    --------
    ndarray
        Processed and reconstructed CT slice
    """
    # Reduce pore contrast and get specimen info
    slice_data, mask, edge_of_specimen, max_radius, min_radius, cluster_means = reduce_pore_contrast(
        slice_data, reduce_contrast=True)

    # Add artifacts
    slice_data = add_edge_cupping_artefact(slice_data, mask, edge_of_specimen)
    slice_data = create_rings(slice_data, 80, max_radius, mask)
    slice_data = create_large_ring(slice_data, min_radius, max_radius, mask)

    # Create and process sinogram
    angles = np.linspace(0, 180, 800)
    sinogram = radon(slice_data, theta=angles, circle=True)

    sinogram_no_stripes = remove_stripe_based_sorting(sinogram, 200, 1)
    sinogram_no_stripes = cp.transpose(sinogram_no_stripes)
    sinogram_no_stripes = cp.asnumpy(sinogram_no_stripes)

    # Reconstruct image
    reconstructed = iradon(sinogram_no_stripes, theta=angles,
                           filter_name='shepp-logan',
                           interpolation='cubic')

    # Scale to 16-bit range
    reconstructed = (reconstructed - np.min(reconstructed)) / (
            np.max(reconstructed) - np.min(reconstructed)) * 65535

    return reconstructed.astype(np.uint16)


def test_middle_slice(input_folder, output_folder):
    """
    Test the processing pipeline on the middle slice of the first H5 file.

    Parameters:
    -----------
    input_folder : str
        Path to folder containing input H5 files
    output_folder : str
        Path to folder where test results will be saved
    """
    h5_files = [f for f in os.listdir(input_folder) if f.endswith('.h5')]
    if not h5_files:
        print("No H5 files found in input folder")
        return

    test_file = h5_files[0]
    print(f'Testing processing pipeline on middle slice of: {test_file}')

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    input_path = os.path.join(input_folder, test_file)

    with h5py.File(input_path, 'r') as f:
        middle_idx = f['data'].shape[0] // 2
        print(f'Extracting middle slice (index {middle_idx}) from total {f["data"].shape[0]} slices')

        slice_data = f['data'][middle_idx]
        original_slice = slice_data.copy()

        print("Processing slice...")
        try:
            processed_slice = process_single_slice(slice_data)

            # Save test outputs
            save_test_outputs(output_folder, original_slice, slice_data, processed_slice)

        except Exception as e:
            print(f'Error during processing: {str(e)}')
            raise
        finally:
            cp.get_default_memory_pool().free_all_blocks()

    print('Test processing completed!')