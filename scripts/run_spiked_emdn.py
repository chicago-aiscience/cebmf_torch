import argparse
import abc
import dataclasses
import logging
import math
import os
import pathlib
from typing import Protocol

import dotenv
import matplotlib.pyplot as plt
import torch

from cebmf_torch.cebnm.cash_solver import cash_posterior_means
from cebmf_torch.cebnm.cov_gb_prior import cgb_posterior_means
from cebmf_torch.cebnm.cov_sharp_gb_prior import sharp_cgb_posterior_means
from cebmf_torch.cebnm.emdn import emdn_posterior_means, EmdnPosteriorMeanNorm
from cebmf_torch.cebnm.spiked_emdn import spiked_emdn_posterior_means

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)s %(message)s',
                    datefmt='%Y-%m-%dT%H:%M:%S',
                    level=logging.INFO)

@dataclasses.dataclass
class Config:
    plots_output_dir: pathlib.Path
    profile_output_dir: pathlib.Path
    prior_name: str
    env_file: pathlib.Path
    hidden_dim: int
    num_gaussians: int
    num_layers: int
    num_samples: int
    num_epochs: int
    penalty: float
    seed: int
    device: torch.device

    def __post_init__(self):
        self.plots_output_dir.mkdir(parents=True, exist_ok=True)
        self.profile_output_dir.mkdir(parents=True, exist_ok=True)

@dataclasses.dataclass
class Prior(abc.ABC):
    """Abstract base class for prior distributions.

    This defines the interface that all prior implementations must follow.
    """
    profile_output_dir: pathlib.Path

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Return the name of the prior distribution."""
        ...

    @abc.abstractmethod
    def posterior_means(
        self,
        X: torch.Tensor,
        betahat: torch.Tensor,
        sebetahat: torch.Tensor,
        n_epochs: int = 50,
        n_layers: int = 4,
        hidden_dim: int = 64,
        n_gaussians: int = 5,
        **kwargs,
    ) -> EmdnPosteriorMeanNorm:
        """Compute posterior means for the given data.

        Parameters
        ----------
        X : torch.Tensor
            Covariates for each observation, shape (n_samples, n_features).
        betahat : torch.Tensor
            Observed effect estimates, shape (n_samples,).
        sebetahat : torch.Tensor
            Standard errors of the effect estimates, shape (n_samples,).
        n_epochs : int, optional
            Number of training epochs (default=50).
        n_layers : int, optional
            Number of hidden layers in the neural network (default=4).
        hidden_dim : int, optional
            Number of hidden units in each layer (default=64).
        n_gaussians : int, optional
            Number of Gaussian components in the mixture (default=5).
        **kwargs
            Additional keyword arguments specific to the prior implementation.

        Returns
        -------
        EmdnPosteriorMeanNorm
            Container with posterior means, standard deviations, and model parameters.
        """
        ...

@dataclasses.dataclass
class EMDN(Prior):
    """Empirical Bayes Normal Means with Mixture Density Network prior.

    This prior models the distribution as a mixture of Gaussians without a spike.
    """

    def __post_init__(self) -> None:
        """Validate the EMDN configuration."""
        pass

    @property
    def name(self) -> str:
        """Return the name of the prior distribution."""
        return "emdn"

    def posterior_means(
        self,
        X: torch.Tensor,
        betahat: torch.Tensor,
        sebetahat: torch.Tensor,
        n_epochs: int = 50,
        n_layers: int = 4,
        hidden_dim: int = 64,
        n_gaussians: int = 5,
        device: torch.device | None = None,
        **kwargs,
    ) -> EmdnPosteriorMeanNorm:
        """Compute posterior means using EMDN prior."""
        # EMDN doesn't have penalty parameter, so we don't pass it
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        ) as prof:
            with torch.profiler.record_function("emdn_posterior_means"):
                res = emdn_posterior_means(
                    X=X,
                    betahat=betahat,
                    sebetahat=sebetahat,
                    n_epochs=n_epochs,
                    n_layers=n_layers,
                    n_gaussians=n_gaussians,
                    hidden_dim=hidden_dim,
                    device=device,
                )
        try:
            record_profile(prof, self.name, self.profile_output_dir)
        except Exception as e:
            logging.warning(f"Failed to record profile: {e}")
        return res

@dataclasses.dataclass
class SpikedEMDN(Prior):
    """Empirical Bayes Normal Means with Spiked Mixture Density Network prior.

    This prior models the distribution as a mixture of Gaussians plus a point mass at zero.
    """

    penalty: float = dataclasses.field(default=1.0) # Penalty parameter for spiked EMDN (>1 encourages spike)

    def __post_init__(self) -> None:
        """Validate the SpikedEMDN configuration."""
        if self.penalty <= 0:
            raise ValueError("penalty must be positive")

    @property
    def name(self) -> str:
        """Return the name of the prior distribution."""
        return "spiked_emdn"

    def posterior_means(
        self,
        X: torch.Tensor,
        betahat: torch.Tensor,
        sebetahat: torch.Tensor,
        n_epochs: int = 50,
        n_layers: int = 4,
        hidden_dim: int = 64,
        n_gaussians: int = 5,
        penalty: float | None = None,
        device: torch.device | None = None,
        **kwargs,
    ) -> EmdnPosteriorMeanNorm:
        """Compute posterior means using Spiked EMDN prior."""
        if penalty is None:
            penalty = self.penalty

        # penalty is keyword-only in spiked_emdn_posterior_means (after *)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        ) as prof:
            with torch.profiler.record_function("spiked_emdn_posterior_means"):
                res =  spiked_emdn_posterior_means(
                    X=X,
                    betahat=betahat,
                    sebetahat=sebetahat,
                    n_epochs=n_epochs,
                    n_layers=n_layers,
                    n_gaussians=n_gaussians,
                    hidden_dim=hidden_dim,
                    penalty=penalty,  # Must be keyword argument
                    device=device,
                )
        try:
            record_profile(prof, self.name, self.profile_output_dir)
        except Exception as e:
            logging.warning(f"Failed to record profile: {e}")
        return res

@dataclasses.dataclass
class CASH(Prior):
    """Covariate Adaptive Shrinkage prior."""

    penalty: float = dataclasses.field(default=1.5)
    """Penalty parameter for CASH."""

    def __post_init__(self) -> None:
        """Validate the CASH configuration."""
        if self.penalty <= 0:
            raise ValueError("penalty must be positive")

    @property
    def name(self) -> str:
        """Return the name of the prior distribution."""
        return "cash"

    def posterior_means(
        self,
        X: torch.Tensor,
        betahat: torch.Tensor,
        sebetahat: torch.Tensor,
        n_epochs: int = 50,
        n_layers: int = 4,
        hidden_dim: int = 64,
        n_gaussians: int = 5,  # CASH uses num_classes, but we map n_gaussians to it
        penalty: float | None = None,
        device: torch.device | None = None,
    ) -> EmdnPosteriorMeanNorm:
        """Compute posterior means using CASH prior."""
        if penalty is None:
            penalty = self.penalty

        # CASH uses num_classes instead of n_gaussians, and penalty is positional before device
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        ) as prof:
            with torch.profiler.record_function("cash_posterior_means"):
                res = cash_posterior_means(
                    X=X,
                    betahat=betahat,
                    sebetahat=sebetahat,
                    n_epochs=n_epochs,
                    n_layers=n_layers,
                    num_classes=n_gaussians,  # Map n_gaussians to num_classes
                    hidden_dim=hidden_dim,
                    penalty=penalty,
                    device=device,
        )
        try:
            record_profile(prof, self.name, self.profile_output_dir)
        except Exception as e:
            logging.warning(f"Failed to record profile: {e}")
        return res

@dataclasses.dataclass
class CGB(Prior):
    """Covariate Generalized-Binary prior."""

    penalty: float = dataclasses.field(default=1.5)
    """Penalty parameter for CGB."""

    def __post_init__(self) -> None:
        """Validate the CGB configuration."""
        if self.penalty <= 0:
            raise ValueError("penalty must be positive")

    @property
    def name(self) -> str:
        """Return the name of the prior distribution."""
        return "cgb"

    def posterior_means(
        self,
        X: torch.Tensor,
        betahat: torch.Tensor,
        sebetahat: torch.Tensor,
        n_epochs: int = 50,
        n_layers: int = 4,
        hidden_dim: int = 64,
        n_gaussians: int = 5,  # CGB doesn't use n_gaussians, but we accept it for interface consistency
        penalty: float | None = None,
        device: torch.device | None = None,
    ) -> EmdnPosteriorMeanNorm:
        """Compute posterior means using CGB prior."""
        if penalty is None:
            penalty = self.penalty

        # CGB has penalty as positional before model_param
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        ) as prof:
            with torch.profiler.record_function("cgb_posterior_means"):
                res = cgb_posterior_means(
                    X=X,
                    betahat=betahat,
                    sebetahat=sebetahat,
                    n_epochs=n_epochs,
                    n_layers=n_layers,
                    hidden_dim=hidden_dim,
                    penalty=penalty,
                    device=device,
                )
        try:
            record_profile(prof, self.name, self.profile_output_dir)
        except Exception as e:
            logging.warning(f"Failed to record profile: {e}")
        return res

@dataclasses.dataclass
class CGBSharp(Prior):
    """Covariate Generalized-Binary sharp prior."""

    penalty: float = dataclasses.field(default=1.5)
    """Penalty parameter for CGBSharp."""

    def __post_init__(self) -> None:
        """Validate the CGBSharp configuration."""
        if self.penalty <= 0:
            raise ValueError("penalty must be positive")

    @property
    def name(self) -> str:
        """Return the name of the prior distribution."""
        return "cgb_sharp"

    def posterior_means(
        self,
        X: torch.Tensor,
        betahat: torch.Tensor,
        sebetahat: torch.Tensor,
        n_epochs: int = 50,
        n_layers: int = 4,
        hidden_dim: int = 64,
        n_gaussians: int = 5,  # CGBSharp doesn't use n_gaussians, but we accept it for interface consistency
        penalty: float | None = None,
        device: torch.device | None = None,
    ) -> EmdnPosteriorMeanNorm:
        """Compute posterior means using CGBSharp prior."""
        if penalty is None:
            penalty = self.penalty

        # CGBSharp has penalty as positional before model_param
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        ) as prof:
            with torch.profiler.record_function("sharp_cgb_posterior_means"):
                res = sharp_cgb_posterior_means(
                X=X,
                betahat=betahat,
                sebetahat=sebetahat,
                n_epochs=n_epochs,
                n_layers=n_layers,
                hidden_dim=hidden_dim,
                penalty=penalty,
                device=device,
            )
        try:
            record_profile(prof, self.name, self.profile_output_dir)
        except Exception as e:
            logging.warning(f"Failed to record profile: {e}")
        return res

def create_args() -> argparse.ArgumentParser:
    """Create an ArgumentParser from the Config dataclass."""
    parser = argparse.ArgumentParser(description="Profile the posterior means of the spiked EMDN")
    parser.add_argument("--env-file", type=pathlib.Path, default=pathlib.Path(".env"), help="Path to environment file")
    parser.add_argument("--prior-name", type=str, default="spiked_emdn", choices=["emdn", "spiked_emdn", "cgb_sharp", "cash", "cgb", "cgb_sharp"], help="Prior distribution to profile with")
    parser.add_argument("--profile-output-dir", type=pathlib.Path, default=pathlib.Path("./profiles"), help="Path to output directory where profiling results will be saved")
    parser.add_argument("--plots-output-dir", type=pathlib.Path, default=pathlib.Path("./plots"), help="Path to output directory where plots will be saved")
    return parser

def create_config(args: argparse.Namespace, device: torch.device) -> Config:
    """Create a Config object from the arguments.

    Parameters
    ----------
    args : argparse.Namespace
        The arguments from the command line.
    device : torch.device
        The device to use for the computation.
    Returns
    -------
    Config
        The configuration object.
    """
    return Config(
        plots_output_dir=args.plots_output_dir,
        profile_output_dir=args.profile_output_dir,
        prior_name=args.prior_name,
        env_file=args.env_file,
        hidden_dim=int(os.environ.get('HIDDEN_DIM', 64)),
        num_gaussians=int(os.environ.get('NUM_GAUSSIANS', 5)),
        num_layers=int(os.environ.get('NUM_LAYERS', 4)),
        num_samples=int(os.environ.get('NUM_SAMPLES', 20_000)),
        num_epochs=int(os.environ.get('NUM_EPOCHS', 50)),
        penalty=float(os.environ.get('PENALTY', 1.0)),
        seed=int(os.environ.get('SEED', 1)),
        device=device,
    )

def print_setup(cfg: Config):
    """Print the setup of the configuration.

    Parameters
    ----------
    cfg : Config
        The configuration object.
    """
    logging.info("Prior distribution: %s", cfg.prior_name)
    logging.info("Profile output directory: %s", cfg.profile_output_dir)
    logging.info("Plots output directory: %s", cfg.plots_output_dir)
    logging.info("Environment file: %s", cfg.env_file)
    logging.info("Seed: %s", cfg.seed)
    logging.info("Number of samples: %s", cfg.num_samples)
    logging.info("Number of epochs: %s", cfg.num_epochs)
    logging.info("Number of layers: %s", cfg.num_layers)
    logging.info("Number of Gaussians: %s", cfg.num_gaussians)
    logging.info("Hidden dimension: %s", cfg.hidden_dim)
    logging.info("Penalty: %s", cfg.penalty)
    logging.info("Device: %s", cfg.device)

def get_data(cfg: Config) -> tuple[torch.Tensor, torch.Tensor]:
    """Get the data for the profiling."""
    # Generate data in PyTorch
    n_samples = cfg.num_samples
    y = torch.empty(n_samples).uniform_(-0.5, 2.5)        # U(-0.5, 2.5)
    X_covariate = y.view(-1, 1)    # 1 sample, 1 column --> 1 feature

    # masks for zero vs nonzero regions
    mask_zero = ((y > 0) & (y < 0.5)) | ((y > 1.5) & (y < 2.0))

    # noise level depends on x
    noise_std = 0.5 + torch.abs(torch.sin(math.pi * y))

    # xtrue: 0 in masked regions, Gaussian elsewhere with that std
    xtrue = torch.where(mask_zero, torch.zeros_like(y), torch.randn_like(y) * noise_std)

    # Observed data
    sobs = torch.ones_like(y)  # known noise level
    xobs = xtrue + torch.randn_like(y) * sobs
    X = xobs.view(-1, 1)

    return X_covariate, xtrue, xobs, sobs, y

def plot_initial_distributions(y: torch.Tensor, xtrue: torch.Tensor, xobs: torch.Tensor, out_path: pathlib.Path):
    """Plot the initial distributions of the data."""
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    colors = torch.where(xtrue == 0, torch.tensor(0), torch.tensor(1))  # 0/1 labels
    axs[0].scatter(y.cpu().numpy(), xtrue.cpu().numpy(),
                c=['green' if c == 0 else 'blue' for c in colors.tolist()],
                s=4, alpha=0.6)
    axs[0].set_title('True Underlying Effect')
    axs[0].set_xlabel('y')
    axs[0].set_ylabel('xtrue')

    axs[1].scatter(y.cpu().numpy(), xobs.cpu().numpy(),
                c=['green' if c == 0 else 'blue' for c in colors.tolist()],
                s=4, alpha=0.6)
    axs[1].set_title('Observed Data')
    axs[1].set_xlabel('y')
    axs[1].set_ylabel('xobs')

    plt.tight_layout()
    plt.savefig(out_path / "initial_distributions.png")

def create_prior(name: str, penalty: float = 1.0, profile_output_dir: pathlib.Path | None = None) -> Prior:
    """Factory function to create a Prior instance from its name.

    Parameters
    ----------
    name : str
        Name of the prior distribution. Must be one of: "emdn", "spiked_emdn", "cgb_sharp", "cash", "cgb".
    penalty : float, optional
        Penalty parameter for priors that support it (>1 encourages spike). Default is 1.0.
    profile_output_dir : pathlib.Path, optional
        The directory to save the profile. If None, defaults to "./profiles".

    Returns
    -------
    Prior
        An instance of the requested prior distribution.

    Raises
    ------
    ValueError
        If the prior name is not recognized.
    """
    if profile_output_dir is None:
        profile_output_dir = pathlib.Path("./profiles")

    if name == "emdn":
        return EMDN(profile_output_dir=profile_output_dir)
    elif name == "spiked_emdn":
        return SpikedEMDN(penalty=penalty, profile_output_dir=profile_output_dir)
    elif name == "cgb_sharp":
        return CGBSharp(penalty=penalty, profile_output_dir=profile_output_dir)
    elif name == "cash":
        return CASH(penalty=penalty, profile_output_dir=profile_output_dir)
    elif name == "cgb":
        return CGB(penalty=penalty, profile_output_dir=profile_output_dir)
    else:
        raise ValueError(
            f"Unknown prior name: {name}. "
            f"Supported priors: 'emdn', 'spiked_emdn', 'cgb_sharp', 'cash', 'cgb'"
        )

def record_profile(prof: torch.profiler.profile, prior_name: str, profile_output_dir: pathlib.Path):
    """Record the profile of the model fitting process using torch.profiler.

    Parameters
    ----------
    prof: torch.profiler.profile
        The profiler object.
    prior_name: str
        The name of the prior.
    profile_output_dir: pathlib.Path
        The directory to save the profile.
    """
    summary_file = profile_output_dir / f"summary_{prior_name}.txt"
    key_averages = prof.key_averages()
    with open(summary_file, "w") as f:
        f.write(f"Profiling Summary for '{prior_name}'\n")
        f.write("="*80 + "\n\n")
        f.write("Top 50 operations by CPU time:\n")
        f.write(key_averages.table(sort_by="cpu_time_total", row_limit=50))
        # Add memory statistics if available
        try:
            mem_table = key_averages.table(sort_by="self_cpu_memory_usage", row_limit=50)
            f.write("\n\nTop 50 operations by memory usage:\n")
            f.write(mem_table)
        except (AttributeError, KeyError):
            pass  # Memory profiling not available
    logging.info(f"Detailed summary exported to: {summary_file}")

def plot_posterior_means(y: torch.Tensor, xobs: torch.Tensor, xtrue: torch.Tensor, posterior_means: EmdnPosteriorMeanNorm, prior_name: str, out_dir: pathlib.Path):
    """Plot the posterior means."""
    # Plot true vs posterior means
    plt.figure(figsize=(15,6))
    plt.scatter(y, xobs, alpha=0.3, label=r"$x_{\rm obs}$")
    plt.scatter(y, xtrue, c="green", s=3, label=r"$x_{\rm true}$")
    plt.scatter(y, posterior_means.post_mean, c="red", s=3, label=f"{prior_name} posterior mean")
    plt.legend()
    plt.xlabel("y")
    plt.ylabel("Effect")
    plt.savefig(out_dir / f"posterior_means_{prior_name}.png")

def main():
    """Main function to profile the posterior means."""
    # Set up configuration and device operations
    arg_parser = create_args()
    args = arg_parser.parse_args()
    dotenv.load_dotenv(args.env_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = create_config(args, device)
    print_setup(cfg)

    # Retrieve data and plot initial distributions
    logging.info("Generating data...")
    torch.manual_seed(cfg.seed)
    X_covariate, xtrue, xobs, sobs, y = get_data(cfg)
    plot_initial_distributions(y, xtrue, xobs, cfg.plots_output_dir)
    logging.info("Generating data completed.")

    # Select the model via the prior name
    logging.info("Creating prior...")
    prior = create_prior(cfg.prior_name, cfg.penalty, cfg.profile_output_dir)
    logging.info("Creating prior completed: %s.", prior.name)

    # Compute posterior means
    logging.info("Profiling posterior means computation...")
    posterior_means = prior.posterior_means(X_covariate, xobs, sobs, n_epochs=cfg.num_epochs, n_layers=cfg.num_layers, hidden_dim=cfg.hidden_dim, n_gaussians=cfg.num_gaussians, device=cfg.device)
    logging.info("Profiling posterior means computation completed.")

    # Plot posterior means
    logging.info("Plotting posterior means...")
    plot_posterior_means(y, xobs, xtrue, posterior_means, prior.name, cfg.plots_output_dir)
    logging.info("Plotting posterior means completed.")


if __name__ == "__main__":
    main()
