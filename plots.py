import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


# Plot the samples for Player 1 (s1) and Player 2 (s2) on the same plot
def plot_posterior_samples(samples_s1, samples_s2):
    plt.figure(figsize=(12, 6))

    plt.plot(samples_s1, label="Samples of s1 (Player 1)", color="blue")
    plt.plot(samples_s2, label="Samples of s2 (Player 2)", color="orange")

    # Set the title and labels
    plt.title("Posterior Distribution Samples for s1 and s2 (If player 1 Wins)")
    plt.xlabel("Iteration")
    plt.ylabel("Skill Value")
    plt.legend()

    # plt.savefig("samples.png", dpi=300, bbox_inches="tight")

    # Display the plot
    plt.grid(True)
    plt.show()


# Plot the convergence of skill means and standard deviations
def plot_convergence(
    s1_running_mean, s2_running_mean, s1_uncertainty, s2_uncertainty, burnin
):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot running means of s1 and s2 (convergence)
    ax1.plot(s1_running_mean, label="Skill 1 Mean", color="blue")
    ax1.plot(s2_running_mean, label="Skill 2 Mean", color="orange")
    ax1.axvline(burnin, color="red", linestyle="--", label="Burn-in Period")
    ax1.set_title("Convergence of Skill Means")
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Mean of Skills")
    ax1.legend()

    # Plot uncertainties for s1 and s2
    ax2.set_title("Convergence of Skill Standard Deviation")
    ax2.set_xlabel("Iterations")
    ax2.set_ylabel("Standard Deviation of Skills")
    ax2.plot(s1_uncertainty, label="Skill 1 Uncertainty", color="blue")
    ax2.plot(s2_uncertainty, label="Skill 2 Uncertainty", color="orange")
    ax2.axvline(burnin, color="red", linestyle="--", label="Burn-in Period")
    plt.tight_layout()

    # plt.savefig("convergence.png", dpi=300, bbox_inches="tight")

    plt.show()


def plot_histogram_gauss(
    samples_s1, samples_s2, mean_s1, mean_s2, burnin, K, elapsed_time
):
    std_s1 = np.std(samples_s1[burnin:])
    std_s2 = np.std(samples_s2[burnin:])

    # Create a range of x values for the Gaussian fit
    x_s1 = np.linspace(mean_s1 - 3 * std_s1, mean_s1 + 3 * std_s1, 100)
    x_s2 = np.linspace(mean_s2 - 3 * std_s2, mean_s2 + 3 * std_s2, 100)

    # Define the Gaussian PDF for s1 and s2
    gaussian_fit_s1 = norm.pdf(x_s1, mean_s1, std_s1)
    gaussian_fit_s2 = norm.pdf(x_s2, mean_s2, std_s2)

    plt.figure(figsize=(10, 6))

    # Plot both s1 and s2 samples and Gaussian fits on the same plot
    plt.hist(
        samples_s1[burnin:],
        bins=30,
        density=True,
        label="s1 Samples",
        color="blue",
        alpha=0.6,
    )
    plt.hist(
        samples_s2[burnin:],
        bins=30,
        density=True,
        label="s2 Samples",
        color="orange",
        alpha=0.6,
    )
    plt.plot(x_s1, gaussian_fit_s1, label="Gaussian Fit for s1", color="blue")
    plt.plot(x_s2, gaussian_fit_s2, label="Gaussian Fit for s2", color="orange")

    # Set title and labels
    plt.title(f"Histograms of {K} samples for s1 and s2 after burn-in")
    plt.xlabel("Skill Level")
    plt.ylabel("Density")
    plt.legend()

    plt.tight_layout()

    # Save the merged plot
    # plt.savefig(f"merged_hists-{K}.png", dpi=300, bbox_inches="tight")

    plt.show()
    print("Time taken for", K, "samples:", elapsed_time)


# Q4.d - plot prior and posterior distributions of s1 and s2
def plot_prior_and_posterior(mu_s1, sig1, samples_s1, samples_s2, burnin):
    # Mean from posterior samples for s1
    mu_s1_post = np.mean(samples_s1[burnin:])
    # Std from posterior samples for s1
    sig_s1_post = np.std(samples_s1[burnin:])

    # Mean from posterior samples for s2
    mu_s2_post = np.mean(samples_s2[burnin:])
    # Std from posterior samples for s2
    sig_s2_post = np.std(samples_s2[burnin:])

    # Plotting range (use common x-axis since their prior is the same)
    x = np.linspace(mu_s1 - 3 * sig1, mu_s1 + 3 * sig1, 100)

    # Create the plot with both prior and posteriors on the same axis
    plt.figure(figsize=(10, 6))

    # Plot the prior for both s1 and s2 (since their prior is the same)
    plt.plot(x, norm.pdf(x, mu_s1, sig1), label="Prior of s1 and s2", color="orange")

    # Plot the posterior of s1
    plt.plot(
        x, norm.pdf(x, mu_s1_post, sig_s1_post), label="Posterior of s1", color="blue"
    )

    # Plot the posterior of s2
    plt.plot(
        x,
        norm.pdf(x, mu_s2_post, sig_s2_post),
        label="Posterior of s2",
        color="deeppink",
    )

    # Add labels and title
    plt.title("Posterior and Prior Distributions of s1 and s2")
    plt.xlabel("s1, s2")
    plt.ylabel("Probability Density")

    # Add legend
    plt.legend()

    # Show plot
    plt.tight_layout()
    # plt.savefig("prior-posterior.png", dpi=300, bbox_inches="tight")
    plt.show()


# Plot moment matching vs gibbs sampling
def plot_mm_gibbs(samples_s1, samples_s2, s1_dist, s2_dist):
    plt.figure(figsize=(10, 6))

    # Player 1
    x_values_s1 = np.linspace(min(samples_s1), max(samples_s1), 100)
    pdf_posterior_s1 = s1_dist.pdf(x_values_s1)

    gibbs_fit_s1 = norm(loc=np.mean(samples_s1), scale=np.std(samples_s1))
    pdf_gibbs_fit_s1 = gibbs_fit_s1.pdf(x_values_s1)

    # Plot the posterior PDF from message passing for Player 1
    plt.plot(
        x_values_s1,
        pdf_posterior_s1,
        label="Posterior from message passing (s1)",
        color="blue",
        lw=2,
    )

    # Plot the histogram of Gibbs samples for Player 1
    plt.hist(
        samples_s1,
        bins=30,
        density=True,
        alpha=0.6,
        color="orange",
        label="Histogram from Gibbs sampling (s1)",
    )

    # Plot the Gaussian approximation fitted to Gibbs samples for Player 1
    plt.plot(
        x_values_s1,
        pdf_gibbs_fit_s1,
        label="Gaussian fit to Gibbs samples (s1)",
        color="red",
        lw=2,
        linestyle="dashed",
    )

    # Player 2
    x_values_s2 = np.linspace(min(samples_s2), max(samples_s2), 100)
    pdf_posterior_s2 = s2_dist.pdf(x_values_s2)

    gibbs_fit_s2 = norm(loc=np.mean(samples_s2), scale=np.std(samples_s2))
    pdf_gibbs_fit_s2 = gibbs_fit_s2.pdf(x_values_s2)

    # Plot the posterior PDF from message passing for Player 2
    plt.plot(
        x_values_s2,
        pdf_posterior_s2,
        label="Posterior from message passing (s2)",
        color="green",
        lw=2,
    )

    # Plot the histogram of Gibbs samples for Player 2
    plt.hist(
        samples_s2,
        bins=30,
        density=True,
        alpha=0.4,
        color="purple",
        label="Histogram from Gibbs sampling (s2)",
    )

    # Plot the Gaussian approximation fitted to Gibbs samples for Player 2
    plt.plot(
        x_values_s2,
        pdf_gibbs_fit_s2,
        label="Gaussian fit to Gibbs samples (s2)",
        color="brown",
        lw=2,
        linestyle="dashed",
    )

    plt.title(
        "Posterior Comparison for Player 1 and Player 2: Message Passing vs Gibbs Sampling",
        fontsize=16,
    )
    plt.xlabel("Skill Level", fontsize=14)
    plt.ylabel("Probability Density", fontsize=14)
    plt.legend()

    plt.tight_layout()
    # plt.savefig("mm-gibbs-combined.png", dpi=300, bbox_inches="tight")
    plt.show()


# Plots the probability density functions of teams' skill distributions
def plot_team_distribution(teams):
    plt.figure(figsize=(10, 6))
    for team, stats in teams.items():
        mu = stats["mu"]
        sig = stats["sig"]
        # Generate normal distribution based on mu and sig
        x = np.linspace(mu - 3 * sig, mu + 3 * sig, 100)
        y = (1 / (sig * np.sqrt(2 * np.pi))) * np.exp(-((x - mu) ** 2) / (2 * sig**2))

        plt.plot(x, y, label=team)

    plt.title("Distribution of Teams After ADF")
    plt.xlabel("Mean of Skills")
    plt.ylabel("Probability Density")
    plt.legend(loc="best")
    # plt.savefig("rugby-teams.png", dpi=300, bbox_inches="tight")
    plt.show()
