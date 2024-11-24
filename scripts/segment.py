# scripts/segment.py
import os
import argparse
from data_worsener.segmentation.segment_h5 import process_h5_files, test_middle_slice


def parse_arguments():
    """Parse command line arguments for segmentation."""
    parser = argparse.ArgumentParser(
        description='Segment CT images and save as H5 files.'
    )

    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input folder containing H5 files'
    )

    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output folder for segmented files'
    )

    parser.add_argument(
        '--chunks', '-c',
        type=int,
        default=8,
        help='Number of chunks to process the data in (default: 8)'
    )

    parser.add_argument(
        '--test-only',
        action='store_true',
        help='Only run the test on middle slice'
    )

    return parser.parse_args()


def validate_paths(input_folder):
    """Validate input path."""
    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist")
        return False

    h5_files = [f for f in os.listdir(input_folder) if f.endswith('.h5')]
    if not h5_files:
        print(f"Error: No H5 files found in input folder '{input_folder}'")
        return False

    return True


def main():
    """Main entry point for segmentation workflow."""
    args = parse_arguments()

    if not validate_paths(args.input):
        return

    try:
        if args.test_only:
            print("\n=== Testing segmentation pipeline on single slice ===")
            test_middle_slice(args.input, args.output)
            return

        process_h5_files(args.input, args.output, args.chunks)
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    except Exception as e:
        print(f"\nError during processing: {str(e)}")
        raise
    finally:
        print("\nSegmentation completed!")


if __name__ == "__main__":
    main()