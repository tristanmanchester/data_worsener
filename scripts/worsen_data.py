# scripts/worsen_data.py
import os
import argparse
from data_worsener.utils import process_h5_files, test_middle_slice


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Process CT images by adding artifacts and reconstructing.'
    )

    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input folder containing H5 files'
    )

    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output folder for processed files'
    )

    parser.add_argument(
        '--test-only',
        action='store_true',
        help='Only run the test on middle slice'
    )

    parser.add_argument(
        '--skip-test',
        action='store_true',
        help='Skip the test and directly process all files'
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='Force processing without asking for confirmation'
    )

    return parser.parse_args()


def validate_paths(input_folder, output_folder):
    """Validate input and output paths."""
    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist")
        return False

    h5_files = [f for f in os.listdir(input_folder) if f.endswith('.h5')]
    if not h5_files:
        print(f"Error: No H5 files found in input folder '{input_folder}'")
        return False

    if not os.path.exists(output_folder):
        try:
            os.makedirs(output_folder)
            print(f"Created output directory: {output_folder}")
        except Exception as e:
            print(f"Error creating output directory: {str(e)}")
            return False

    return True


def main():
    """Main entry point."""
    args = parse_arguments()

    if not validate_paths(args.input, args.output):
        return

    try:
        if not args.skip_test:
            print("\n=== Testing processing pipeline on single slice ===")
            test_middle_slice(args.input, args.output)

        if args.test_only:
            return

        if args.force or args.skip_test or input(
                "\nDo you want to proceed with processing all files? (y/n): ").lower() == 'y':
            print("\n=== Processing all H5 files ===")
            process_h5_files(args.input, args.output)
        else:
            print("Full processing skipped.")

    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    except Exception as e:
        print(f"\nError during processing: {str(e)}")
        raise
    finally:
        print("\nProcessing completed!")


if __name__ == "__main__":
    main()