import numpy as np
import pandas as pd
from scipy.stats import norm, truncnorm, multivariate_normal
import random
import time
from plots import *


# GIBBS SAMPLING FUNCTIONS
# Function for generating samples from truncated normal distribution
def normal_truncated(a, b, mu, sig):
    # Transform the bounds to z-scores using (bound-mu)/sig
    a = (a - mu) / sig
    b = (b - mu) / sig
    return truncnorm.rvs(a, b, loc=mu, scale=sig)


# Q4 - Gibbs sampling function for estimating p(s1,s2|y)
def gibbs_sampler(mu_s1, mu_s2, sig1, sig2, beta, M, y, K, consider_draw):

    # Initialize samples_s1 and samples_s2
    samples_s1 = np.zeros(K)
    samples_s2 = np.zeros(K)

    # Initialize s1 and s2 by drawing random samples from (mu_1,sig1) and (mu_2,sig2)
    s1 = np.random.normal(mu_s1, sig1)
    s2 = np.random.normal(mu_s2, sig2)

    samples_s1[0] = s1
    samples_s2[0] = s2

    mu = np.array([mu_s1, mu_s2]).T
    cov = np.array([[sig1**2, 0], [0, sig2**2]])

    for i in range(1, K):
        mu_t = samples_s1[i - 1] - samples_s2[i - 1]

        if y == 0 and not consider_draw:
            continue

        # Sample t first from p(t|s1,s2,y) -> Q3.b
        if y == 1:
            # a = 0, b = inf
            t = normal_truncated(0, np.inf, mu_t, beta)
        elif y == -1:
            # a = -inf, b = 0
            t = normal_truncated(-np.inf, 0, mu_t, beta)
        elif consider_draw and y == 0:
            # for draw
            t = 0

        # Sample s1 and s2 from t -> Q3.a
        # Calculate new variance
        a = np.linalg.inv(cov)
        b = (1 / (beta**2)) * np.outer(M, M)
        var_hat = np.linalg.inv(a + b)

        # Calculate new mean
        c = np.matmul(a, mu)
        d = (1 / (beta**2)) * M * t
        mu_hat = np.matmul(var_hat, (c + d))

        # Sample from the multivariate gaussian distribution
        samples_s1[i], samples_s2[i] = multivariate_normal.rvs(mu_hat, var_hat, 1)

    return samples_s1, samples_s2


# Function for transforming samples from gibbs samples into gauss distributions
def gauss_approximation(s1, s2):
    # Calculate the mean of samples
    mean_s1 = np.mean(s1)
    mean_s2 = np.mean(s2)

    # Calculate the covariance of samples
    cov = np.cov(s1, s2)

    gauss_approx = multivariate_normal(mean=[mean_s1, mean_s2], cov=cov)

    return mean_s1, mean_s2, cov, gauss_approx


# Q4.c - Function for running gibbs sampling with various number of iterations to choose the best value for K
def compare_gibbs_iterations(mu_s1, mu_s2, sig1, sig2, beta, M, y, burnin):
    iterations = [5000, 8000, 10000, 20000]

    for i in iterations:
        start_time = time.time()
        samples_s1, samples_s2 = gibbs_sampler(
            mu_s1, mu_s2, sig1, sig2, beta, M, y, i, False
        )

        # Decided to choose 8000 as the best value for K
        if i == 8000:
            samples_s1_final, samples_s2_final = samples_s1, samples_s2

        # Calculate how much time the sampling takes for each iteration
        elapsed_time = time.time() - start_time

        mean_s1, mean_s2, cov, gauss_approx = gauss_approximation(
            samples_s1[burnin:], samples_s2[burnin:]
        )
        # Plot histograms of samples with gaussian fit after burn-in
        plot_histogram_gauss(
            samples_s1[burnin:],
            samples_s2[burnin:],
            mean_s1,
            mean_s2,
            burnin,
            i,
            elapsed_time,
        )

    return samples_s1_final, samples_s2_final


# PREDICTION FUNCTIONS
# Function for predicting the outcome of a match based on the mu and sig of skills
def predict(mu_s1, mu_s2, sig1, sig2):
    mu = mu_s1 - mu_s2
    sig = np.sqrt(sig1**2 + sig2**2)
    # Calculate the probability of team2 winning using CDF
    team2_win_prob = norm.cdf(0, loc=mu, scale=sig)

    # If the probability that team 2 wins is more than 50%, predict team 2 wins else team 1 wins
    if team2_win_prob > 0.5:
        return -1
    else:
        return 1


# Function for predicting the outcome of a match randomly
def random_predict():
    return random.choice([1, -1])


# Function for predicting the outcome of a game when we consider draws
def predict_with_draw(mu_s1, mu_s2, sig1, sig2):
    mu = mu_s1 - mu_s2
    sig = np.sqrt(sig1**2 + sig2**2)
    team2_win_prob = norm.cdf(0, loc=mu, scale=sig)

    # If the probability that team 2 wins is more than 55%, predict team 2 wins
    if team2_win_prob > 0.55:
        return -1  # Team 2 wins
    # If the probability that team 1 wins (1 - team2_win_prob) is greater than 55%, predict team 1 wins
    elif (1 - team2_win_prob) > 0.55:
        return 1  # Team 1 wins
    # Otherwise, predict a draw (return 0)
    else:
        return 0


# ADF FUNCTIONS
# Q5 - Function for Assumed Density Filtering - without considering draws
def ADF(df, mu_s1, mu_s2, sig1, sig2, beta, K, burnin, M, toPredict, randomGuess):

    # Dictionary to keep track of teams and their mu and sig
    team_stats = {}

    correct_guesses = 0
    total_guesses = 0

    # Iterate over the rows of the dataframe
    for i, (index, row) in enumerate(df.iterrows()):
        team1 = row["team1"]
        team2 = row["team2"]
        score1 = row["score1"]
        score2 = row["score2"]

        y = score1 - score2

        # ignore draw
        if y == 0:
            continue

        y = np.sign(y)

        # Get mu and sig for the teams (if not found, set to previously defined values (mu_s1 = mu_s2 = 25, sig_s1 = sig_s2 = 25/3)
        mu_s1, sig1 = team_stats.get(team1, {"mu": mu_s1, "sig": sig1}).values()
        mu_s2, sig2 = team_stats.get(team2, {"mu": mu_s2, "sig": sig2}).values()

        # If prediction is required
        if toPredict:
            # If making a random prediction is required
            if randomGuess:
                predicted_y = random_predict()
            else:
                predicted_y = predict(mu_s1, mu_s2, sig1, sig2)

            # If predicted y is correct, add to the number of correct predictions
            if predicted_y == y:
                correct_guesses += 1
            total_guesses += 1

        # Gibbs sampling to update the posterior distribution of the teams
        team1_samples, team2_samples = gibbs_sampler(
            mu_s1, mu_s2, sig1, sig2, beta, M, y, K, consider_draw=False
        )

        # Update the mu and sig of the teams
        mu_s1_new = np.mean(team1_samples[burnin:])
        sig1_new = np.std(team1_samples[burnin:])

        mu_s2_new = np.mean(team2_samples[burnin:])
        sig2_new = np.std(team2_samples[burnin:])

        # Store the updated mu and sig in the team_stats
        team_stats[team1] = {"mu": mu_s1_new, "sig": sig1_new}
        team_stats[team2] = {"mu": mu_s2_new, "sig": sig2_new}

        if i > 0 and i % 100 == 0:
            print(f"Processing {i} rows is finished.")

    if toPredict:
        return team_stats, correct_guesses, total_guesses
    else:
        return team_stats


# Function for Assumed Density Filtering - with considering draws
def ADF_with_draw(
    df, mu_s1, mu_s2, sig1, sig2, beta, K, burnin, M, toPredict, randomGuess
):

    team_stats = {}
    # Dictionary to keep track of priors for each team
    team_priors = {}

    correct_guesses = 0
    total_guesses = 0

    for i, (index, row) in enumerate(df.iterrows()):
        team1 = row["team1"]
        team2 = row["team2"]
        score1 = row["score1"]
        score2 = row["score2"]

        y = score1 - score2

        y = np.sign(y)

        mu_s1, sig1 = team_stats.get(team1, {"mu": mu_s1, "sig": sig1}).values()
        mu_s2, sig2 = team_stats.get(team2, {"mu": mu_s2, "sig": sig2}).values()

        # Initialize prior list for team1 if not already
        if team1 not in team_priors:
            team_priors[team1] = []

        # Initialize prior list for team2 if not already
        if team2 not in team_priors:
            team_priors[team2] = []

        # Store prior for team1
        team_priors[team1].append((mu_s1, sig1))

        # Store prior for team2
        team_priors[team2].append((mu_s2, sig2))

        if toPredict:
            if randomGuess:
                predicted_y = random_predict()
            else:
                predicted_y = predict_with_draw(mu_s1, mu_s2, sig1, sig2)

            if predicted_y == y:
                correct_guesses += 1
            total_guesses += 1

        team1_samples, team2_samples = gibbs_sampler(
            mu_s1, mu_s2, sig1, sig2, beta, M, y, K, consider_draw=True
        )

        # If y is either 1 or -1 (win or loss), calculate new mu and sig for s1 and s2
        if y == 1 or y == -1:
            mu_s1_new = np.mean(team1_samples[burnin:])
            sig1_new = np.std(team1_samples[burnin:])

            mu_s2_new = np.mean(team2_samples[burnin:])
            sig2_new = np.std(team2_samples[burnin:])

        # If y is 0 (draw), calculate new mu and sig for s1 and s2
        elif y == 0:
            # Set the new mu for s1 and s2 as the average of their original mu values
            mu_s1_new = mu_s2_new = (mu_s1 + mu_s2) / 2
            # Set sig for s1 and s2 slightly downward (by 2%) to show a reduction in uncertainty after a draw
            sig1_new = sig1 * 0.98
            sig_2_new = sig2 * 0.98

        team_stats[team1] = {"mu": mu_s1_new, "sig": sig1_new}
        team_stats[team2] = {"mu": mu_s2_new, "sig": sig2_new}

        if i > 0 and i % 100 == 0:
            print(f"Processing {i} rows is finished.")

    if toPredict:
        return team_stats, correct_guesses, total_guesses
    else:
        return team_stats


# Function for calculating
def calculate_prediction_rate(correct_guesses, total_guesses):
    print(f"Correct Guesses: {correct_guesses}")
    print(f"Total Guesses: {total_guesses}")
    pred_rate = correct_guesses / total_guesses
    print(f"Prediction Rate: {pred_rate * 100:.4f}%")


# Function for saving rankings in a .txt file
def save_rankings(sorted_team_stats, filename):
    with open(filename, "w") as file:
        # Write the header
        file.write(f"{'Team':<20}{'Mean of Skills':<20}{'Std':<20}\n")
        file.write("=" * 60 + "\n")
        # Write each team's stats in a formatted text format
        for team, stats in sorted_team_stats.items():
            file.write(f"{team:<20}{stats['mu']:<20}{stats['sig']:<20}\n")


# MOMENT MATCHING FUNCTIONS
# Function for calculating the mean and variance of a truncated Gaussian distribution
def truncated_gaussian(a, b, mu, var):
    a = (a - mu) / np.sqrt(var)
    b = (b - mu) / np.sqrt(var)

    mean = truncnorm.mean(a, b, loc=mu, scale=np.sqrt(var))
    variance = truncnorm.var(a, b, loc=mu, scale=np.sqrt(var))

    return mean, variance


# Function for multiplying two Gaussian distributions
def multipy_gauss(m1, s1, m2, s2):
    s = 1 / (1 / s1 + 1 / s2)
    m = (m1 / s1 + m2 / s2) * s
    # return the mean and variance of the product of two Gaussians
    return m, s


# Function for dividing two Gaussian distributions
def division_gauss(mu_s1, mu_s2, var1, var2):
    return multipy_gauss(mu_s1, mu_s2, var1, -var2)


# Function for moment-matching to update means and variances for both s1 and s2
def moment_matching(mu_s1, mu_s2, sig1, sig2, beta, y):
    # Compute the messages
    m1_m = mu_s1
    m1_var = sig1**2

    m2_m = mu_s2
    m2_var = sig2**2

    m3_m = m1_m
    m3_var = m1_var

    m4_m = m2_m
    m4_var = m2_var

    # Message from factor f_t_s1_s2 to t
    m5_m = m1_m - m2_m
    m5_var = m1_var + m2_var + beta**2

    # Includes m_6, m_7 and m_8
    if y == 1:
        # a = 0, b = +inf
        t_m, t_var = truncated_gaussian(0, np.inf, m5_m, m5_var)
    elif y == -1:
        # a = -inf, b = 0
        t_m, t_var = truncated_gaussian(-np.inf, 0, m5_m, m5_var)

    # Message from t to f_t_s1_s2
    m9_m, m9_var = division_gauss(t_m, t_var, m5_m, m5_var)

    # Update mean and variance of s1
    # Message from f_t_s1_s2 to s1
    m10_m = m9_m + m4_m
    m10_var = m9_var + m4_var + beta**2
    new_m_s1, new_var_s1 = multipy_gauss(m1_m, m1_var, m10_m, m10_var)

    # Update mean and variance of s2
    # Message from f_t_s1_s2 to s2
    m11_m = m4_m - m9_m
    m11_var = m9_var + m3_var + beta**2
    new_m_s2, new_var_s2 = multipy_gauss(m2_m, m2_var, m11_m, m11_var)

    return new_m_s1, new_var_s1, new_m_s2, new_var_s2


# Preproess new dataset (Rugby Dataset)
def preprocess(df):
    selected_columns = ["date", "home_team", "away_team", "home_score", "away_score"]
    df = df[selected_columns].copy()
    # create 'year' column to use for filtering data
    df["year"] = df["date"].str.split("-").str[0]
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    # Include only matches from the year 2015 and later
    df = df[df["year"] >= 2015]
    # rename the columns
    df = df.rename(
        columns={
            "home_team": "team1",
            "away_team": "team2",
            "home_score": "score1",
            "away_score": "score2",
        }
    )
    # drop unncessary columns
    df = df.drop(["date", "year"], axis=1)
    return df
