"""Command-line interface for SubCellPortable."""

import argparse
import sys
from typing import Optional


def create_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser.

    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description="SubCellPortable: Run SubCell model inference on microscopy images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="""
Examples:
  python process.py                            # Use config.yaml settings
  python process.py -o ./results               # Specify output directory
  python process.py -o ./results -g 0 -b 256   # Use GPU 0 with batch size 256
  python process.py -o ./results --embeddings_only  # Generate only embeddings (faster)
  python process.py --config custom.yaml --path_list experiment1.csv  # Custom config and input

Common mistakes:
  ✗ python process.py 2                  # Missing flag for GPU
  ✓ python process.py -g 2               # Correct: use -g for GPU ID

  ✗ python process.py output/            # Missing -o flag
  ✓ python process.py -o output/         # Correct: use -o for output directory

For more help: python process.py --help
Configuration file: Edit config.yaml for easier parameter management
""",
    )

    # Input/config files
    parser.add_argument(
        "--config",
        help="Path to configuration YAML file (default: config.yaml)",
        default=argparse.SUPPRESS,
        type=str,
    )
    parser.add_argument(
        "--path_list",
        help="Path to input CSV file with image paths (default: path_list.csv)",
        default=argparse.SUPPRESS,
        type=str,
    )

    # Model configuration
    parser.add_argument(
        "-c",
        "--model_channels",
        help="Model channel configuration to use. 'auto' selects the best available model per row.",
        choices=["auto", "rybg", "rbg", "ybg", "bg"],
        default=argparse.SUPPRESS,
        type=str,
    )
    parser.add_argument(
        "-m",
        "--model_type",
        help="Model type to be used",
        choices=["mae_contrast_supcon_model", "vit_supcon_model"],
        default=argparse.SUPPRESS,
        type=str,
    )
    parser.add_argument(
        "-u",
        "--update_model",
        action="store_true",
        help="Download/update the selected model files (default: False)",
    )

    # Output configuration
    parser.add_argument(
        "-o",
        "--output_dir",
        help="Output directory for all results (required for new CSV format without output_folder column)",
        default=argparse.SUPPRESS,
        type=str,
    )
    parser.add_argument(
        "-csv",
        "--create_csv",
        action="store_true",
        help="Generate a combined CSV of probabilities and embeddings (default: False)",
    )
    parser.add_argument(
        "--embeddings_only",
        action="store_true",
        help="Only generate embeddings, skip classification (faster)",
    )
    parser.add_argument(
        "--output_format",
        choices=["individual", "combined"],
        default=argparse.SUPPRESS,
        help="Output format: individual (.npy files) or combined (.h5ad file)",
    )
    parser.add_argument(
        "--save_attention_maps",
        action="store_true",
        help="Save attention map images (default: False)",
    )

    # Performance configuration
    parser.add_argument(
        "-g",
        "--gpu",
        help="The GPU id to use [0, 1, 2, 3...]. -1 for CPU usage",
        default=argparse.SUPPRESS,
        type=int,
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        help="Batch size for processing",
        default=argparse.SUPPRESS,
        type=int,
    )
    parser.add_argument(
        "-w",
        "--num_workers",
        help="Number of workers for data loading",
        default=argparse.SUPPRESS,
        type=int,
    )
    parser.add_argument(
        "-p",
        "--prefetch_factor",
        help="Prefetch factor for data loading",
        default=argparse.SUPPRESS,
        type=int,
    )
    parser.add_argument(
        "--async_saving",
        action="store_true",
        help="Save individual files asynchronously (only for individual format, not combined; default: False)",
    )

    # Logging
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress verbose logging (quiet mode)",
    )

    return parser


def parse_args(args: Optional[list] = None) -> dict:
    """Parse command-line arguments.

    Only returns arguments that were explicitly provided by the user.
    This ensures CLI defaults don't override config file values.

    Args:
        args: List of arguments to parse. If None, uses sys.argv

    Returns:
        Dictionary of only explicitly provided arguments
    """
    parser = create_parser()

    # Only parse if there are actual arguments beyond the script name
    if args is None:
        args = sys.argv[1:]

    if len(args) == 0:
        # No arguments provided, return empty dict to use defaults
        return {}

    parsed = parser.parse_args(args)
    result = vars(parsed)

    # Filter out store_true arguments that are False (not provided)
    # These have default False when not specified, but we don't want them
    # to override config file values
    filtered_result = {}
    for key, value in result.items():
        # Keep all non-boolean values (SUPPRESS handles those)
        # Only keep boolean values if True (meaning flag was provided)
        if not isinstance(value, bool) or value is True:
            filtered_result[key] = value

    return filtered_result
