"""
Plot benchmark results from GPU performance diagnostics.

This script creates visualizations of execution time and data size
versus input size from benchmark data.
"""

import argparse
import json
import pathlib
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd


# Constants
FIGURE_SIZE = (14, 6)
DPI = 150
FONT_SIZE_LABEL = 12
FONT_SIZE_TITLE = 14
FONT_SIZE_ANNOTATION = 9
FONT_SIZE_SUPTITLE = 10
LINE_WIDTH = 2
MARKER_SIZE = 8
GRID_ALPHA = 0.3
SUPTITLE_Y_POSITION = 0.98
ANNOTATION_OFFSET = (0, 10)
INVALID_VALUE = -1

HARDWARE_INFO = "NVIDIA A100 GPU with 80GB Memory (requested 16 CPUs and 32GB Memory on Node)"
DEFAULT_INPUT_JSON = pathlib.Path("/net/scratch/ntebaldi/cebmf-data/benchmark/gpu_performance_diagnostic.json")
DEFAULT_OUTPUT_DIR = pathlib.Path("/net/scratch/ntebaldi/cebmf-data/benchmark")
OUTPUT_FILENAME = "benchmark_plots.png"


def load_benchmark_json(input_json: pathlib.Path = DEFAULT_INPUT_JSON) -> pd.DataFrame:
    """Load benchmark data from JSON file into a pandas DataFrame.

    Args:
        input_json: Path to JSON file with benchmark data

    Returns:
        DataFrame with columns: input_size, exe_time, data_size, and other metrics
    """
    with open(input_json, "r") as f:
        data = json.load(f)

    # Convert nested dictionary to list of dictionaries
    # Each key (input_size) becomes a row with all nested values
    records = []
    for input_size_str, metrics in data.items():
        record = {"input_size": int(input_size_str)}
        record.update(metrics)
        records.append(record)

    df = pd.DataFrame(records)

    # Rename columns for plotting compatibility
    # Map JSON column names to expected DataFrame column names
    column_mapping = {
        "gpu_time": "exe_time",  # Use GPU time as execution time
        "data_size_mb": "data_size"  # Use MB as data_size
    }
    df = df.rename(columns=column_mapping)

    # Sort by input_size for proper plotting order
    df = df.sort_values("input_size").reset_index(drop=True)

    return df


def filter_valid_data(df: pd.DataFrame) -> pd.DataFrame:
    """Filter out invalid entries from dataframe.

    Args:
        df: Input dataframe

    Returns:
        Filtered dataframe with only valid entries (non-null and positive values)
    """
    # Filter out rows where exe_time is null or <= 0
    # Also filter out rows where data_size is null or <= 0
    mask = (
        df['exe_time'].notna() &
        (df['exe_time'] > 0) &
        df['data_size'].notna() &
        (df['data_size'] > 0)
    )
    return df[mask].copy()


def configure_axis(
    ax: plt.Axes,
    xlabel: str,
    ylabel: str,
    title: str,
    use_log_scale: bool = True
) -> None:
    """Configure axis labels, title, and scale.

    Args:
        ax: Matplotlib axis to configure
        xlabel: X-axis label
        ylabel: Y-axis label
        title: Plot title
        use_log_scale: Whether to use log scale for both axes
    """
    ax.set_xlabel(xlabel, fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel(ylabel, fontsize=FONT_SIZE_LABEL)
    ax.set_title(title, fontsize=FONT_SIZE_TITLE, fontweight='bold')
    ax.grid(True, alpha=GRID_ALPHA)

    if use_log_scale:
        ax.set_xscale('log')
        ax.set_yscale('log')


def add_value_annotations(
    ax: plt.Axes,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    format_str: str
) -> None:
    """Add value annotations to plot points.

    Args:
        ax: Matplotlib axis
        df: DataFrame with data
        x_col: Column name for x values
        y_col: Column name for y values
        format_str: Format string for annotation (e.g., '{:.1f}s')
    """
    for idx, row in df.iterrows():
        ax.annotate(
            format_str.format(row[y_col]),
            (row[x_col], row[y_col]),
            textcoords="offset points",
            xytext=ANNOTATION_OFFSET,
            ha='center',
            fontsize=FONT_SIZE_ANNOTATION
        )


def plot_execution_time(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Plot execution time vs input size.

    Args:
        ax: Matplotlib axis
        df: DataFrame with benchmark data
    """
    ax.plot(
        df['input_size'],
        df['exe_time'],
        'o-',
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE
    )
    configure_axis(
        ax,
        'Input Size (elements)',
        'Execution Time (seconds)',
        'Execution Time vs Input Size'
    )
    add_value_annotations(ax, df, 'input_size', 'exe_time', '{:.1f}s')


def plot_data_size(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Plot data size vs input size.

    Args:
        ax: Matplotlib axis
        df: DataFrame with benchmark data
    """
    ax.plot(
        df['input_size'],
        df['data_size'],
        's-',
        color='green',
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE
    )
    configure_axis(
        ax,
        'Input Size (elements)',
        'Data Size (MB)',
        'Data Size vs Input Size'
    )
    add_value_annotations(ax, df, 'input_size', 'data_size', '{:.1f}MB')


def create_benchmark_plots(
    df: pd.DataFrame,
    output_dir: pathlib.Path = DEFAULT_OUTPUT_DIR
) -> pathlib.Path:
    """Create and save benchmark plots.

    Args:
        df: DataFrame with benchmark data
        output_dir: Directory to save plots
        show_plots: Whether to display plots

    Returns:
        Path to saved plot file
    """
    # Filter valid data
    df_valid = filter_valid_data(df)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGURE_SIZE)

    # Add hardware information as figure-level label
    fig.suptitle(
        HARDWARE_INFO,
        fontsize=FONT_SIZE_SUPTITLE,
        y=SUPTITLE_Y_POSITION,
        style='italic',
        color='gray'
    )

    # Create plots
    plot_execution_time(ax1, df_valid)
    plot_data_size(ax2, df_valid)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, SUPTITLE_Y_POSITION])

    # Save figure
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / OUTPUT_FILENAME
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')

    plt.close()

    return output_path


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Plot benchmark results from GPU performance diagnostics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--input-json",
        type=pathlib.Path,
        default=DEFAULT_INPUT_JSON,
        help="Path to JSON file with benchmark data"
    )

    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save output plots"
    )

    return parser.parse_args()


def main():
    """Main function to create benchmark plots."""
    args = parse_arguments()

    # Validate input file exists
    if not args.input_json.exists():
        raise FileNotFoundError(f"Input JSON file not found: {args.input_json}")

    # Load data and create plots
    df = load_benchmark_json(args.input_json)
    output_path = create_benchmark_plots(
        df,
        output_dir=args.output_dir
    )
    print(f"Plots saved to: {output_path}")


if __name__ == "__main__":
    main()