import dataclasses
import enum
import pathlib
import typing

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

from cebmf_torch import cEBMF

class PriorBackend(str, enum.Enum):
    """Valid prior backend options."""
    NONE = "None"
    CGB = "cgb"
    CASH = "cash"
    EMDN = "emdn"
    CGB_SHARP = "cgb_sharp"

@dataclasses.dataclass
class TiledClustering:
    """Tiled clustering model.

    Args:
        data: The data to cluster.
        covariate_matrix: The covariate matrix.
        factors: The factors.
        loadings: The loadings.
        prior_backend: The prior backend.
        niter: The number of iterations.
        device: The device to run on.
    """
    data: torch.Tensor | None = None   # (N, P) - observations x features
    covariate_matrix: torch.Tensor | None = None

    factors: torch.Tensor | None = None
    loadings: torch.Tensor | None = None

    prior_backend: str = "None"
    factor_labels: torch.Tensor | None = None

    niter: int = 10
    device: torch.device = torch.device("cpu")

    # Computed attributes (not in __init__)
    model: cEBMF | None = dataclasses.field(init=False, default=None)

    # Constants
    DEFAULT_LOADINGS_BOUNDARIES: typing.ClassVar[tuple[float, float]] = (0.33, 0.66)
    DEFAULT_NOISE_STD: typing.ClassVar[float] = 1.0

    def __post_init__(self):
        # Generate default data if not provided
        if self.data is None:
            # Set default dimensions for generating synthetic data
            self.num_observations = 1000  # Default: 1000 observations
            self.num_features = 100  # Default: 100 features
            self._generate_default_data()

        # Now we know data exists, get dimensions from actual data
        self.num_observations, self.num_features = self.data.shape  # (N, P)

        # Load data to device
        self.data = self.data.to(self.device)
        if self.covariate_matrix is not None:
            self.covariate_matrix = self.covariate_matrix.to(self.device)
        if self.factors is not None:
            self.factors = self.factors.to(self.device)
        if self.loadings is not None:
            self.loadings = self.loadings.to(self.device)
        if self.factor_labels is not None:
            self.factor_labels = self.factor_labels.to(self.device)

    def _generate_default_data(self):
        """Generate default data for the model."""
        # Random uniform data
        x = torch.rand(self.num_observations)
        y = torch.rand(self.num_observations)
        self.covariate_matrix = torch.stack([x, y], dim=1)  # (N, 2) -- kept for clarity/optionally used later

        # Generate factors
        t1 = torch.randint(0, 2, (self.num_features,), dtype=torch.float32)  # {0,1}
        t2 = torch.randint(0, 2, (self.num_features,), dtype=torch.float32)  # {0,1}

        f0 = t1 * torch.randn(self.num_features)
        f1 = t2 * torch.randn(self.num_features)
        f2 = t2 * torch.randn(self.num_features)
        f  = torch.stack([f0, f1, f2], dim=0)  # (3, M)
        self.factors = f

        # Generate loadings
        L = torch.zeros(self.num_observations, 3, dtype=torch.float32)

        b1, b2 = self.DEFAULT_LOADINGS_BOUNDARIES
        mask1 = x < b1
        mask2 = (~mask1) & (x < b2) & (y < b1)  # second region
        mask3 = ~(mask1 | mask2)                # everything else

        # Nonzero loadings by region (note: original logic used sin(x) for all three)
        L[mask1, 0] = torch.sin(x[mask1])
        L[mask2, 1] = torch.sin(x[mask2])
        L[mask3, 2] = torch.sin(x[mask3])
        self.loadings = L

        # Generate factor labels - in {1,2,3}
        factor = torch.zeros(self.num_observations, dtype=torch.long)
        factor[mask1] = 1
        factor[mask2] = 2
        factor[mask3] = 3
        self.factor_labels = factor

        # Generate observations
        noise = self.DEFAULT_NOISE_STD * torch.randn(self.num_observations, self.num_features)
        Z = L @ f + noise  # (N, M)

        # Sanity checks
        assert L.shape == (self.num_observations, 3)
        assert f.shape == (3, self.num_features)
        assert Z.shape == (self.num_observations, self.num_features)
        self.data = Z

    def fit_model(self):
        """Fit the model."""

        if self.prior_backend == "None":
            mycebmf = cEBMF(data=self.data, device=self.device)

        else:
            # For learned priors, need covariates or self-covariates
            if self.covariate_matrix is None:
                # Option 1: Use self-covariates (other factors)
                mycebmf = cEBMF(
                    data=self.data,
                    prior_L=self.prior_backend,
                    self_row_cov=True,  # Use other factors as covariates
                    allow_backfitting=False,
                    device=self.device
                )

            else:
                # Option 2: Use provided covariates
                mycebmf = cEBMF(
                    data=self.data,
                    X_l=self.covariate_matrix,
                    prior_L=self.prior_backend,
                    allow_backfitting=False,
                    device=self.device
                )

        mycebmf.initialise_factors()
        mycebmf.fit(self.niter)

        self.model = mycebmf

    def plot_model(self, output_dir: pathlib.Path, plot_name: str = None) -> pathlib.Path:
        """Plot the model's objective (ELBO) vs iteration.

        Args:
            output_dir: The directory to save the plot.
            plot_name: The name of the plot.
        Returns:
            The path to the saved plot.
        """
        if plot_name is None:
            plot_name = f"model_{self.prior_backend}.png"
        output_path = output_dir / plot_name

        plt.figure(figsize=(8, 5))
        obj_values = _to1d(self.model.obj)
        label = "Point Laplace" if self.prior_backend == "None" else self.prior_backend.replace("_", " ").upper()

        plt.plot(obj_values, label=label)
        plt.title("Objective (ELBO) vs Iteration")
        plt.xlabel("Iteration")
        plt.ylabel("Objective (ELBO)")
        plt.legend(title="Method", loc="best")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return output_path

    def plot_factors_visualization(self, output_dir: pathlib.Path, plot_name: str = None) -> pathlib.Path:
        """Plot the factors visualization.

        Args:
            output_dir: The directory to save the plot.
            plot_name: The name of the plot.
        Returns:
            The path to the saved plot.
        """
        if plot_name is None:
            plot_name = f"factors_visualization_{self.prior_backend}.png"
        output_path = output_dir / plot_name

        # Check if covariate_matrix exists
        if self.covariate_matrix is None:
            raise ValueError("covariate_matrix is required for factor visualization")

        if self.factor_labels is None:
            raise ValueError("factor_labels is required for factor visualization")

        # Extract x and y coordinates from covariate_matrix
        # covariate_matrix shape is (N, D) where D >= 2 for x, y coordinates
        x = self.covariate_matrix[:, 0]  # First column (x coordinates)
        y = self.covariate_matrix[:, 1]  # Second column (y coordinates)

        # Convert to numpy for plotting
        x_np = x.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()

        # Compute boundaries based on actual coordinate medians
        # These match the factor label assignment: regions are split at medians
        x_median = np.median(x_np)  # Longitude median (separates east/west)
        y_median = np.median(y_np)  # Latitude median (separates north/south)

        colors = ["#D41159", "#1A85FF", "#40B0A6"]  # red, blue, teal
        color_map = {1: colors[0], 2: colors[1], 3: colors[2]}
        point_colors = [color_map[int(k)] for k in self.factor_labels.tolist()]

        plt.figure(figsize=(7.5, 6))
        plt.scatter(x_np, y_np, c=point_colors, s=18, alpha=0.85)

        # Draw boundaries: horizontal line at y_median (separates north/south),
        # vertical line at x_median (separates east/west)
        plt.axhline(y_median, color="black", linestyle="--", linewidth=1.5, alpha=0.7)
        plt.axvline(x_median, color="black", linestyle="--", linewidth=1.5, alpha=0.7)
        plt.title("Factor assignment by region")
        plt.xlabel("Longitude (normalized)")
        plt.ylabel("Latitude (normalized)")

        # Legend: combine factor colors and boundary lines
        factor_handles = [
            mpatches.Patch(color=colors[0], label="Factor 1 (Northern regions)"),
            mpatches.Patch(color=colors[1], label="Factor 2 (Southern-Eastern regions)"),
            mpatches.Patch(color=colors[2], label="Factor 3 (Southern-Western regions)"),
        ]
        # Add boundary lines to legend
        boundary_handles = [
            plt.Line2D([0], [0], color="black", linestyle="--", linewidth=1.5, alpha=0.7, label="Region boundaries")
        ]
        plt.legend(handles=factor_handles + boundary_handles, title="Regions", loc="best", frameon=True)
        plt.tight_layout()

        plt.savefig(output_path)
        plt.close()

        return output_path

    def plot_factors(self, output_dir: pathlib.Path, plot_name: str = None) -> pathlib.Path:
        """Plot all factors for this model on the same plot.

        Args:
            output_dir: Output directory where plot will be saved
            plot_name: Name of the plot file (default: factors_{prior_backend}.png)
        Returns:
            The path to the saved plot.
        """
        if plot_name is None:
            plot_name = f"factors_{self.prior_backend}.png"
        output_path = output_dir / plot_name

        # Extract x coordinates from covariate_matrix (first column)
        if self.covariate_matrix is not None:
            x = self.covariate_matrix[:, 0]
            x_np = x.detach().cpu().numpy()
        else:
            # Fallback: use observation indices if no coordinates
            x_np = np.arange(self.model.L.shape[0])

        # Get the number of factors
        num_factors = self.model.L.shape[1]

        # Create a single plot with all factors
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot L[:, k] for each factor k
        for k in range(num_factors):
            # Extract L[:, k] for this factor
            L_k = self.model.L[:, k].detach().cpu().numpy()

            # Plot with different colors/labels for each factor
            ax.scatter(x_np, L_k, s=10, label=f"Factor {k+1}", alpha=0.6)

        # Set plot properties
        method_label = "Point Laplace" if self.prior_backend == "None" else self.prior_backend.replace("_", " ").upper()
        ax.set_title(f"Fitted Factors - {method_label}")
        ax.set_xlabel("x coordinate" if self.covariate_matrix is not None else "Observation index")
        ax.set_ylabel("L[:, k]")
        ax.legend(title="Factor", loc="best")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save the plot
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return output_path


def _to1d(a: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convert tensor or array to 1D numpy array."""
    if isinstance(a, torch.Tensor):
        return a.detach().cpu().flatten().numpy()
    return np.asarray(a).ravel()


if __name__ == "__main__":
    """EXAMPLE - Economics data: economic indicators across geographic regions.

    Real-world context:
    - Observations (N): Cities or regions (e.g., 500 cities)
    - Features (P): Economic indicators (GDP, unemployment, inflation, etc.)
    - Covariates: Geographic coordinates (latitude, longitude)
    - Factors: Economic patterns (e.g., industrial, agricultural, service-based)

    Example: World Bank economic data
    - Cities/regions with geographic coordinates
    - Economic indicators over time
    """
    import argparse
    import logging

    parser = argparse.ArgumentParser(description="Run TiledClustering examples")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=pathlib.Path("./output"),
        help="Output directory for plots"
    )
    parser.add_argument(
        "--niter",
        type=int,
        default=20,
        help="Number of iterations to fit the model"
    )
    parser.add_argument(
        "--prior-backend",
        choices=PriorBackend.__members__.values(),
        default=PriorBackend.CGB_SHARP.value,
        help="Prior backend to use"
    )
    args = parser.parse_args()
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)s %(message)s',
                        datefmt='%Y-%m-%dT%H:%M:%S',
                        level=logging.INFO)

    if torch.cuda.is_available():
        torch.cuda.set_device(0)  # Set default CUDA device to index 0
        device = torch.device("cuda:0")  # Use explicit index for consistent comparison
    else:
        device = torch.device("cpu")
    logging.info(f"Selected device: {device}")

    logging.info(f"Generating data...")
    N = 500   # Number of cities/regions
    P = 50    # Number of economic indicators

    # Geographic coordinates (latitude, longitude)
    latitude = torch.rand(N) * 180 - 90   # -90 to 90 degrees
    longitude = torch.rand(N) * 360 - 180  # -180 to 180 degrees

    # Create factor labels based on geographic regions
    # Define regions using latitude/longitude boundaries
    # This creates 3 distinct geographic regions for visualization
    lat_median = latitude.median()
    lon_median = longitude.median()

    # Region 1: Northern regions (high latitude)
    mask1 = latitude >= lat_median

    # Region 2: Southern regions with eastern longitude (low latitude, high longitude)
    mask2 = (latitude < lat_median) & (longitude >= lon_median)

    # Region 3: Southern regions with western longitude (low latitude, low longitude)
    mask3 = (latitude < lat_median) & (longitude < lon_median)

    # Create factor labels tensor
    factor_labels = torch.zeros(N, dtype=torch.long)
    factor_labels[mask1] = 1  # Northern regions
    factor_labels[mask2] = 2  # Southern-Eastern regions
    factor_labels[mask3] = 3  # Southern-Western regions

    # Normalize coordinates
    lat_norm = (latitude - latitude.mean()) / latitude.std()
    lon_norm = (longitude - longitude.mean()) / longitude.std()
    covariate_matrix = torch.stack([lat_norm, lon_norm], dim=1)  # (N, 2)

    # Generate structured economic data with geographic correlation
    # Create true factors that correlate with geography
    K_true = 3  # Number of underlying economic patterns
    L_true = torch.zeros(N, K_true)
    F_true = torch.randn(P, K_true) * 0.5  # Economic indicator patterns

    # Factor 1: Economic pattern active in northern regions
    # (e.g., industrial/developed regions)
    L_true[mask1, 0] = torch.randn(mask1.sum()) * 1.5 + 2.0

    # Factor 2: Economic pattern active in southern-eastern regions
    # (e.g., agricultural/export-oriented regions)
    L_true[mask2, 1] = torch.randn(mask2.sum()) * 1.5 + 2.0

    # Factor 3: Economic pattern active in southern-western regions
    # (e.g., service/tourism regions)
    L_true[mask3, 2] = torch.randn(mask3.sum()) * 1.5 + 2.0

    # Generate economic indicators from factors + noise
    # This creates data where geographic regions have distinct economic profiles
    noise = torch.randn(N, P) * 0.8  # Moderate noise
    data = L_true @ F_true.T + noise

    # Normalize the data (mean=0, std=1) for better model performance
    data = (data - data.mean()) / data.std()

    logging.info(f"Generated structured data with geographic correlation")
    logging.info(f"  - Factor 1 (Northern): {mask1.sum()} regions")
    logging.info(f"  - Factor 2 (Southern-Eastern): {mask2.sum()} regions")
    logging.info(f"  - Factor 3 (Southern-Western): {mask3.sum()} regions")

    logging.info(f"Creating TiledClustering object...")
    clustering = TiledClustering(
        data=data,
        covariate_matrix=covariate_matrix,
        factor_labels=factor_labels,
        prior_backend=args.prior_backend,
        niter=args.niter,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    logging.info(f"Fitting model...")
    clustering.fit_model()

    logging.info(f"Plotting model...")
    output_path = clustering.plot_model(output_dir, plot_name="model.png")
    logging.info(f"Saved model plot to {output_path}")

    logging.info(f"Plotting factors...")
    output_path = clustering.plot_factors(output_dir, plot_name="factors.png")
    logging.info(f"Saved factors plot to {output_path}")

    logging.info(f"Plotting factors visualization...")
    output_path = clustering.plot_factors_visualization(output_dir, plot_name="factors_visualization.png")
    logging.info(f"Saved factors visualization plot to {output_path}")