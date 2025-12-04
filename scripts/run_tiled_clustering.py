import argparse
import dataclasses
import datetime
import logging
import pathlib
import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

# Add the source directory to Python path if cebmf_torch is not installed
# This allows the script to work with kernprof and when run directly
try:
    from cebmf_torch import cEBMF
except ImportError:
    # Try multiple methods to find the source directory
    _script_file = None
    if '__file__' in globals():
        _script_file = __file__

    elif len(sys.argv) > 0:
        _script_file = sys.argv[0]

    if _script_file:
        _script_dir = pathlib.Path(_script_file).resolve().parent
        _project_root = _script_dir.parent
        _src_dir = _project_root / "src"

        # Try adding src directory
        if _src_dir.exists() and str(_src_dir) not in sys.path:
            sys.path.insert(0, str(_src_dir))

        # Also try adding project root (in case package structure is different)
        if str(_project_root) not in sys.path:
            sys.path.insert(0, str(_project_root))

    # Try import again
    from cebmf_torch import cEBMF

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)s %(message)s',
                    datefmt='%Y-%m-%dT%H:%M:%S',
                    level=logging.INFO)
# file_handler = logging.FileHandler("/net/scratch/ntebaldi/cebmf-data/logs/run-tiled-clustering.log", mode='a')
# file_handler.setLevel(logging.INFO)
# file_handler.setFormatter(logging.Formatter('%(asctime)s,%(msecs)d %(levelname)s %(message)s',
#                                             datefmt='%Y-%m-%dT%H:%M:%S'))
# logging.getLogger().addHandler(file_handler)


@dataclasses.dataclass
class Config:
    in_dir: pathlib.Path = dataclasses.field(default=pathlib.Path("./data"), metadata={
        "help": "Path to input directory containing model prior information"
    })
    out_dir: pathlib.Path = dataclasses.field(default=pathlib.Path("./output"), metadata={
        "help": "Path to output directory where results will be saved"
    })
    seed: int = dataclasses.field(default=1, metadata={
        "help": "Random seed for reproducibility"
    })
    num_obs: int = dataclasses.field(default=2000, metadata={
        "help": "Number of observations"
    })
    num_features: int = dataclasses.field(default=200, metadata={
        "help": "Number of features"
    })
    niter: int = dataclasses.field(default=10, metadata={
        "help": "Number of iterations to fit the model"
    })
    boundaries: tuple[float, float] = dataclasses.field(default=(0.33, 0.66), metadata={
        "help": "Boundaries for the clustering model"
    })
    noise_std: float = dataclasses.field(default=1.0, metadata={
        "help": "Standard deviation of the noise"
    })
    prior_list: list[str] = dataclasses.field(default_factory=lambda: ["None", "cgb_sharp", "cash", "cgb", "emdn"], metadata={
        "help": "List of prior distributions for the factors"
    })
    profile: bool = dataclasses.field(default=False, metadata={
        "help": "Enable CPU profiling of model fitting"
    })
    profile_output_dir: pathlib.Path = dataclasses.field(default=pathlib.Path("./profiles"), metadata={
        "help": "Directory to save profiling results"
    })
    profile_iterations: int = dataclasses.field(default=None, metadata={
        "help": "Number of iterations to profile (default: profile all iterations)"
    })

    def __post_init__(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir = self.out_dir / "plots"
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        if self.profile:
            self.profile_output_dir.mkdir(parents=True, exist_ok=True)


class ListAction(argparse.Action):
    """Custom action to handle list arguments with comma or space separation."""
    def __init__(self, option_strings, dest, default=None, default_factory=None, **kwargs):
        self.default_factory = default_factory
        # Remove default_factory from kwargs before passing to parent
        kwargs.pop('default_factory', None)
        super().__init__(option_strings, dest, default=default or [], **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        # Handle comma-separated string if single value contains commas
        if isinstance(values, list) and len(values) == 1 and isinstance(values[0], str) and ',' in values[0]:
            values = [x.strip() for x in values[0].split(',')]
        setattr(namespace, self.dest, values)


def create_args() -> argparse.ArgumentParser:
    """Create an ArgumentParser from the Config dataclass."""
    parser = argparse.ArgumentParser(description="Run image classifier on organoid images")

    for field in dataclasses.fields(Config):
        # Build argument flag and help message
        flags = [f"--{field.name.replace('_', '-')}"]
        kwargs = {
            "help": field.metadata.get("help", ""),
            "default": field.default
        }

        # Determine argument type
        if field.type == bool:
            kwargs["action"] = "store_true" if field.default is False else "store_false"

        elif field.type == pathlib.Path:
            # Handle pathlib.Path type
            kwargs["type"] = pathlib.Path

        elif hasattr(field.type, '__origin__'):
            if field.type.__origin__ is tuple:
                # Handle tuple types - expect comma-separated values
                element_types = field.type.__args__
                kwargs["type"] = lambda value: _tuple_parser(value, element_types)

            elif field.type.__origin__ is list:
                # Handle list types - accept space or comma-separated values
                kwargs["nargs"] = "*"
                kwargs["action"] = ListAction
                # Store default_factory in action if available
                if field.default_factory is not dataclasses.MISSING:
                    kwargs["default_factory"] = field.default_factory
                    kwargs["default"] = []  # Empty list as default, will be replaced if not provided
                elif field.default is not dataclasses.MISSING:
                    kwargs["default"] = field.default
                else:
                    kwargs["default"] = []

        else:
            kwargs["type"] = field.type

        parser.add_argument(*flags, **kwargs)

    return parser

def _tuple_parser(value: str, element_types: tuple):
    """Parse comma-separated string into a tuple with specified types."""
    parts = [float(x.strip()) for x in value.split(',')]
    if len(parts) != len(element_types):
        raise argparse.ArgumentTypeError(
            f"Expected {len(element_types)} values separated by commas"
        )
    return tuple(parts)

def get_args():
    arg_parser = create_args()
    args = arg_parser.parse_args()
    args_dict = vars(args)

    # Apply default_factory for list fields that weren't provided (empty list case)
    for field in dataclasses.fields(Config):
        if hasattr(field.type, '__origin__') and field.type.__origin__ is list:
            field_name = field.name
            value = args_dict.get(field_name)
            # If empty list (or None) and default_factory exists, use it
            if not value or (isinstance(value, list) and len(value) == 0):
                if field.default_factory is not dataclasses.MISSING:
                    args_dict[field_name] = field.default_factory()

    for key, val in args_dict.items():
        logging.info(f"{key}: {val}")

    cfg = Config(**args_dict)
    return cfg

def generate_data(cfg: Config):
    # Random uniform data
    x = torch.rand(cfg.num_obs)
    y = torch.rand(cfg.num_obs)
    X = torch.stack([x, y], dim=1)  # (N, 2) -- kept for clarity/optionally used later
    plot_scatter(x, y, cfg.plot_dir / "random_uniform_data.png")

    # Generate factors
    t1 = torch.randint(0, 2, (cfg.num_features,), dtype=torch.float32)  # {0,1}
    t2 = torch.randint(0, 2, (cfg.num_features,), dtype=torch.float32)  # {0,1}

    f0 = t1 * torch.randn(cfg.num_features)
    f1 = t2 * torch.randn(cfg.num_features)
    f2 = t2 * torch.randn(cfg.num_features)
    f  = torch.stack([f0, f1, f2], dim=0)  # (3, M)

    # Generate loadings
    L = torch.zeros(cfg.num_obs, 3, dtype=torch.float32)

    b1, b2 = cfg.boundaries
    mask1 = x < b1
    mask2 = (~mask1) & (x < b2) & (y < b1)  # second region
    mask3 = ~(mask1 | mask2)                # everything else

    # Nonzero loadings by region (note: original logic used sin(x) for all three)
    L[mask1, 0] = torch.sin(x[mask1])
    L[mask2, 1] = torch.sin(x[mask2])
    L[mask3, 2] = torch.sin(x[mask3])

    # Generate factor labels - in {1,2,3}
    factor = torch.zeros(cfg.num_obs, dtype=torch.long)
    factor[mask1] = 1
    factor[mask2] = 2
    factor[mask3] = 3
    plot_factor_visualization(x, y, b1, b2, factor, cfg.plot_dir / "factor_visualization.png")
    plot_individual_factors(x, y, b1, b2, L, f, cfg.plot_dir / "individual_factors.png")

    # Generate observations
    noise = cfg.noise_std * torch.randn(cfg.num_obs, cfg.num_features)
    Z = L @ f + noise  # (N, M)

    # Optional sanity checks
    assert L.shape == (cfg.num_obs, 3)
    assert f.shape == (3, cfg.num_features)
    assert Z.shape == (cfg.num_obs, cfg.num_features)

    return Z, X

def fit_models(data: torch.Tensor, X: torch.Tensor, prior_list: list[str], device: torch.device,
               niter: int = 10, profile: bool = False, profile_output_dir: pathlib.Path = None, profile_iterations: int = None):
    """Fit models with optional CPU profiling.

    Args:
        data: Input data tensor
        X: Covariate matrix
        prior_list: List of prior names to fit
        device: Device to run on
        niter: Number of iterations to fit
        profile: Whether to enable profiling
        profile_output_dir: Directory to save profiling results
        profile_iterations: Number of iterations to profile (None = all)
    """
    models = {}
    for prior in prior_list:
        logging.info(f"\n{'='*80}")
        logging.info(f"Fitting model with prior: {prior}")
        logging.info(f"{'='*80}")

        if prior == "None":
            mycebmf = cEBMF(data=data, device=device)
        else:
            mycebmf = cEBMF(data=data,
                            X_l=X,
                            prior_L=prior,
                            allow_backfitting=False,
                            device=device)

        # Verify model tensors are on the correct device
        logging.info("Verifying model tensor devices:")
        verify_tensor_device(mycebmf.Y, f"model.Y ({prior})", device)
        if mycebmf.covariate.X_l is not None:
            verify_tensor_device(mycebmf.covariate.X_l, f"model.X_l ({prior})", device)
        if hasattr(mycebmf, 'L') and mycebmf.L is not None:
            verify_tensor_device(mycebmf.L, f"model.L ({prior})", device)
        if hasattr(mycebmf, 'F') and mycebmf.F is not None:
            verify_tensor_device(mycebmf.F, f"model.F ({prior})", device)

        mycebmf.initialise_factors()

        # Log GPU memory usage before fitting
        if torch.cuda.is_available():
            # Reset peak memory stats to track peak during fitting
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()  # Clear any cached memory

            memory_allocated_before = torch.cuda.memory_allocated() / (1024**2)  # MB
            memory_reserved_before = torch.cuda.memory_reserved() / (1024**2)  # MB
            logging.info(f"GPU memory before fitting: allocated={memory_allocated_before:.2f} MB, reserved={memory_reserved_before:.2f} MB")

        # Profile model fitting if enabled
        if torch.cuda.is_available():
            # Create CUDA events to measure GPU time
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

        if profile:
            profile_model_fitting(mycebmf, prior, niter, profile_output_dir, profile_iterations)
        else:
            mycebmf.fit(niter)

        if torch.cuda.is_available():
            end_event.record()
            torch.cuda.synchronize()  # Wait for GPU operations to complete
            gpu_time_ms = start_event.elapsed_time(end_event)
            logging.info(f"GPU computation time: {gpu_time_ms:.2f} ms ({gpu_time_ms/1000:.2f} seconds)")

        # Log GPU memory usage after fitting
        if torch.cuda.is_available():
            memory_allocated_after = torch.cuda.memory_allocated() / (1024**2)  # MB
            memory_reserved_after = torch.cuda.memory_reserved() / (1024**2)  # MB
            memory_peak_allocated = torch.cuda.max_memory_allocated() / (1024**2)  # MB - peak during fitting
            memory_peak_reserved = torch.cuda.max_memory_reserved() / (1024**2)  # MB - peak reserved

            logging.info(f"GPU memory after fitting: allocated={memory_allocated_after:.2f} MB, reserved={memory_reserved_after:.2f} MB")
            logging.info(f"GPU memory peak during fitting: allocated={memory_peak_allocated:.2f} MB, reserved={memory_peak_reserved:.2f} MB")
            logging.info(f"GPU memory increase: allocated={memory_allocated_after - memory_allocated_before:.2f} MB")
            logging.info(f"GPU memory peak increase: allocated={memory_peak_allocated - memory_allocated_before:.2f} MB")

        models[prior] = mycebmf
        logging.info(f"Fitted model with prior {prior}")

        # Clear GPU cache between models to prevent memory accumulation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            memory_after_model = torch.cuda.memory_allocated() / (1024**2)  # MB
            logging.info(f"GPU memory after model '{prior}': {memory_after_model:.2f} MB")

    return models


def profile_model_fitting(model: cEBMF, prior_name: str, niter: int, profile_output_dir: pathlib.Path, profile_iterations: int = None):
    """Profile the model fitting process using torch.profiler.

    Args:
        model: cEBMF model to fit and profile
        prior_name: Name of the prior (for output file naming)
        niter: Total number of iterations to fit
        profile_output_dir: Directory to save profiling results
        profile_iterations: Number of iterations to profile (None = all)
    """
    # Clear GPU cache before profiling to reduce memory pressure
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        memory_before = torch.cuda.memory_allocated() / (1024**2)  # MB
        memory_reserved = torch.cuda.memory_reserved() / (1024**2)  # MB
        memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**2)  # MB
        memory_available = memory_total - memory_reserved

        logging.info(f"GPU memory before profiling: allocated={memory_before:.2f} MB, reserved={memory_reserved:.2f} MB")
        logging.info(f"GPU memory available: {memory_available:.2f} MB / {memory_total:.2f} MB")

        if memory_available < 500:  # Warn if less than 500 MB available
            logging.warning(f"Low GPU memory available ({memory_available:.2f} MB). Profiling may cause OOM.")

    # Configure profiler activities
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    # Determine how many iterations to profile
    profile_niter = profile_iterations if profile_iterations else niter
    logging.info(f"Profiling model '{prior_name}' for {profile_niter} iterations...")

    # Profile the model fitting
    # Note: record_shapes, profile_memory, and with_stack increase memory usage significantly
    # Disable profile_memory to reduce OOM risk
    with torch.profiler.profile(
        activities=activities,
        record_shapes=False,  # Disable to reduce memory overhead
        profile_memory=False,  # Disable memory profiling to reduce OOM risk
        with_stack=False,  # Disable stack traces to reduce memory overhead
    ) as prof:
        with torch.profiler.record_function(f"model_fitting_{prior_name}"):
            model.fit(profile_niter)

    # Continue fitting remaining iterations without profiling if needed
    if profile_iterations and profile_iterations < niter:
        remaining = niter - profile_iterations
        logging.info(f"Continuing to fit remaining {remaining} iterations without profiling...")
        model.fit(remaining)

    # Clear GPU cache after profiling to free up memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Export profiling results
    logging.info(f"Exporting profiling results for '{prior_name}'...")
    record_profile(prof, prior_name, profile_output_dir)

    # Clear profiler object to free memory
    del prof
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def record_profile(prof: torch.profiler.profile, prior_name: str, profile_output_dir: pathlib.Path):
    """Record the profile of the model fitting process using torch.profiler.

    Args:
        prof: torch.profiler.profile object
        prior_name: Name of the prior (for output file naming)
        profile_output_dir: Directory to save profiling results
    """
    # Print summary table
    print(f"\n{'='*80}")
    print(f"Profiling Results for '{prior_name}'")
    print(f"{'='*80}")
    print("\n--- Summary Table (sorted by CPU time) ---")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
    print(f"{'='*80}\n")

    # Export stack traces (text format)
    stack_file = profile_output_dir / f"stack_traces_{prior_name}.txt"
    with open(stack_file, "w") as f:
        f.write(prof.key_averages(group_by_stack_n=5).table(sort_by="cpu_time_total"))
    logging.info(f"Stack traces exported to: {stack_file}")

    # Export detailed summary table
    summary_file = profile_output_dir / f"summary_{prior_name}.txt"
    with open(summary_file, "w") as f:
        f.write(f"Profiling Summary for '{prior_name}'\n")
        f.write("="*80 + "\n\n")
        f.write("Top 50 operations by CPU time:\n")
        f.write(prof.key_averages().table(sort_by="cpu_time_total", row_limit=50))
        # Add memory statistics if available
        try:
            mem_table = prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=50)
            f.write("\n\nTop 50 operations by memory usage:\n")
            f.write(mem_table)
        except (AttributeError, KeyError):
            pass  # Memory profiling not available
    logging.info(f"Detailed summary exported to: {summary_file}")

def plot_scatter(x: torch.Tensor, y: torch.Tensor, out_path: pathlib.Path):
    plt.figure(figsize=(7, 5))
    plt.scatter(x.numpy(), y.numpy(), alpha=0.5, s=12)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Scatter: x vs y")
    plt.tight_layout()
    plt.savefig(out_path)

def plot_factor_visualization(x: torch.Tensor, y: torch.Tensor, b1:float, b2:float, factor: torch.Tensor, out_path: pathlib.Path):
    colors = ["#D41159", "#1A85FF", "#40B0A6"]  # red, blue, teal
    color_map = {1: colors[0], 2: colors[1], 3: colors[2]}
    point_colors = [color_map[int(k)] for k in factor.tolist()]

    plt.figure(figsize=(7.5, 6))
    plt.scatter(x.numpy(), y.numpy(), c=point_colors, s=18, alpha=0.85)
    for v in (b1, b2):
        plt.axhline(v, color="black", linestyle="--", linewidth=1)
        plt.axvline(v, color="black", linestyle="--", linewidth=1)
    plt.title("Factor assignment by region")
    plt.xlabel("x")
    plt.ylabel("y")

    # Legend (explicit: red = non-zero L[:,0], etc.)
    handles = [
        mpatches.Patch(color=colors[0], label="factor 1 (non-zero L[:,0])"),
        mpatches.Patch(color=colors[1], label="factor 2 (non-zero L[:,1])"),
        mpatches.Patch(color=colors[2], label="factor 3 (non-zero L[:,2])"),
    ]
    plt.legend(handles=handles, title="Groups", loc="best", frameon=True)
    plt.tight_layout()
    plt.savefig(out_path)
    logging.info(f"Saved factor visualization plot to {out_path}")
    plt.close()

def plot_individual_factors(x: torch.Tensor, y: torch.Tensor, b1: float, b2: float, L: torch.Tensor, f: torch.Tensor, out_path: pathlib.Path):
    # Create a single figure with 3 subplots (1 row x 3 columns)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i in range(3):

        ax = axes[i]
        scatter = ax.scatter(
            x.numpy(),
            y.numpy(),
            c=L[:, i].numpy(),
            cmap="coolwarm",  # shows magnitude of L[:,i]
            s=18
        )

        for v in (b1, b2):
            ax.axhline(v, color="black", linestyle="--", linewidth=1)
            ax.axvline(v, color="black", linestyle="--", linewidth=1)

        ax.set_title(f"Factor {i+1} values (L[:,{i}])")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(f"L{i+1}")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved individual factors plot to {out_path}")

def plot_models(models: dict, plot_dir: pathlib.Path):
    """Plot ELBO (objective) values for each fitted model.

    Args:
        models: Dictionary mapping prior names to fitted cEBMF model objects
        plot_dir: Output directory where plots will be saved
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # Iterate through models dictionary and plot each one
    for prior_name, model in models.items():
        # Convert model.obj to 1D numpy array for plotting
        obj_values = _to1d(model.obj)
        # Use "None" as label for EBMF (no prior)
        label = "Point Laplace" if prior_name == "None" else prior_name.replace("_", " ").upper()
        ax.plot(obj_values, label=label)

    ax.set_title("Objective (ELBO) vs Iteration")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Objective (ELBO)")
    ax.legend(title="Method", loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save the plot instead of showing it
    output_path = plot_dir / "elbo_convergence.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved ELBO plot to {output_path}")

def _to1d(a):
    """Convert tensor or array to 1D numpy array."""
    if isinstance(a, torch.Tensor):
        return a.detach().cpu().flatten().numpy()
    return np.asarray(a).ravel()

def plot_factors(models: dict, X: torch.Tensor, plot_dir: pathlib.Path):
    """Plot comparison of fitted factors across all models.

    Args:
        models: Dictionary mapping prior names to fitted cEBMF model objects
        X: Covariate matrix with shape (N, 2) where first column is x coordinates
        plot_dir: Output directory where plots will be saved
    """
    # Extract x coordinates from X (first column)
    x = X[:, 0]
    x_np = x.detach().cpu().numpy()

    # Get the number of factors (use minimum K across all models)
    num_factors = min(model.L.shape[1] for model in models.values())

    # Plot each factor k
    for k in range(num_factors):
        fig, ax = plt.subplots(figsize=(7, 5))

        # Plot L[:, k] for each model
        for prior_name, model in models.items():
            # Get label
            label = "Point Laplace" if prior_name == "None" else prior_name.replace("_", " ").upper()

            # Extract L[:, k] for this model
            L_k = model.L[:, k].detach().cpu().numpy()

            # Plot
            ax.scatter(x_np, L_k, s=10, label=label)

        ax.set_title(f"Fitted Factor {k+1}")
        ax.set_xlabel("x")
        ax.set_ylabel(f"L[:, {k}]")
        ax.legend(title="Method", loc="best")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save the plot
        output_path = plot_dir / f"factor_{k+1}_comparison.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        logging.info(f"Saved factor {k+1} comparison plot to {output_path}")

def log_gpu_info():
    """Log comprehensive GPU information and diagnostics."""
    logging.info("="*80)
    logging.info("GPU DIAGNOSTICS")
    logging.info("="*80)

    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    logging.info(f"CUDA available: {cuda_available}")

    if cuda_available:
        # GPU count and names
        num_gpus = torch.cuda.device_count()
        logging.info(f"Number of GPUs: {num_gpus}")
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            logging.info(f"  GPU {i}: {gpu_name}")

        # Current device
        current_device = torch.cuda.current_device()
        logging.info(f"Current CUDA device index: {current_device}")
        logging.info(f"Current CUDA device name: {torch.cuda.get_device_name(current_device)}")

        # Memory information
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            memory_total = props.total_memory / (1024**3)  # GB
            logging.info(f"  GPU {i} total memory: {memory_total:.2f} GB")

        # Test tensor creation on GPU
        try:
            test_tensor = torch.randn(10, 10, device="cuda")
            logging.info(f"✓ Successfully created test tensor on GPU")
            logging.info(f"  Test tensor device: {test_tensor.device}")
            logging.info(f"  Test tensor location: {test_tensor.device.type}:{test_tensor.device.index if test_tensor.device.index is not None else 'default'}")
            del test_tensor
            torch.cuda.empty_cache()
        except Exception as e:
            logging.error(f"✗ Failed to create tensor on GPU: {e}")
    else:
        logging.info("No CUDA devices available - will use CPU")

    logging.info("="*80)


def verify_tensor_device(tensor: torch.Tensor, name: str, expected_device: torch.device):
    """Verify that a tensor is on the expected device and log the result.

    This function compares device type and index, treating cuda and cuda:0 as equivalent.
    """
    actual_device = tensor.device

    # Normalize devices for comparison (cuda -> cuda:0, etc.)
    # Get actual device index (default to 0 for CUDA if None)
    actual_index = actual_device.index if actual_device.index is not None else 0
    expected_index = expected_device.index if expected_device.index is not None else 0

    # Compare device type and index
    type_match = actual_device.type == expected_device.type
    index_match = actual_index == expected_index

    is_correct = type_match and index_match

    status = "✓" if is_correct else "✗"
    logging.info(f"{status} {name}: device={actual_device}, expected={expected_device}")

    if not is_correct:
        logging.warning(f"  WARNING: {name} is on {actual_device} but expected {expected_device}")

    return is_correct


def main():
    start = datetime.datetime.now()

    # Set up device
    if torch.cuda.is_available():
        torch.cuda.set_device(0)  # Set default CUDA device to index 0
        device = torch.device("cuda:0")  # Use explicit index for consistent comparison
    else:
        device = torch.device("cpu")

    # Log comprehensive GPU information
    log_gpu_info()
    logging.info(f"Selected device: {device}")

    cfg = get_args()

    # Set up random seed
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)
    # torch.backends.cudnn.enabled = False

    # Generate data
    data, X = generate_data(cfg)
    logging.info(f"Generated data")

    # Verify data tensors are on the correct device (cEBMF will move them, but let's check)
    logging.info("Verifying data tensor devices:")
    verify_tensor_device(data, "data", device)
    verify_tensor_device(X, "X", device)

    # Fit the model
    models = fit_models(data, X, cfg.prior_list, device, cfg.niter,
                       profile=cfg.profile,
                       profile_output_dir=cfg.profile_output_dir,
                       profile_iterations=cfg.profile_iterations)
    logging.info(f"Fitted models")

    # Visualize the models
    plot_models(models, cfg.plot_dir)
    plot_factors(models, X, cfg.plot_dir)

    end = datetime.datetime.now()
    logging.info(f"Elapsed time: {end - start}")


if __name__ == "__main__":
    main()