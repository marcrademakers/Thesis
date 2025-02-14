import numpy as np
import scipy.stats as stats
from scikit_posthocs import posthoc_nemenyi_friedman

# Data from the table (Precision, Recall, and F1-score for each setup)
zero_shot = np.array([
    [0.206, 0.255, 0.228],  # Jira
    [0.531, 0.508, 0.519],  # Lyrasis
    [0.326, 0.433, 0.372],  # Network Obs
    [0.360, 0.412, 0.384],  # OpenShift
    [0.592, 0.381, 0.464],  # Qt Design
    [0.354, 0.460, 0.400]   # Red Hat
])

cross_project = np.array([
    [0.297, 0.733, 0.423],  # Jira
    [0.492, 0.733, 0.589],  # Lyrasis
    [0.330, 0.678, 0.444],  # Network Obs
    [0.348, 0.628, 0.448],  # OpenShift
    [0.383, 0.579, 0.461],  # Qt Design
    [0.271, 0.406, 0.325]   # Red Hat
])

project_specific = np.array([
    [0.274, 0.537, 0.363],  # Jira
    [0.483, 0.311, 0.378],  # Lyrasis
    [0.403, 0.651, 0.498],  # Network Obs
    [0.528, 0.704, 0.603],  # OpenShift
    [0.489, 0.657, 0.561],  # Qt Design
    [0.419, 0.617, 0.499]   # Red Hat
])

# Perform the Friedman test
friedman_precision = stats.friedmanchisquare(zero_shot[:, 0], cross_project[:, 0], project_specific[:, 0])
friedman_recall = stats.friedmanchisquare(zero_shot[:, 1], cross_project[:, 1], project_specific[:, 1])
friedman_f1 = stats.friedmanchisquare(zero_shot[:, 2], cross_project[:, 2], project_specific[:, 2])

# Stack data into a format suitable for post-hoc testing
data_precision = np.vstack((zero_shot[:, 0], cross_project[:, 0], project_specific[:, 0])).T
data_recall = np.vstack((zero_shot[:, 1], cross_project[:, 1], project_specific[:, 1])).T
data_f1 = np.vstack((zero_shot[:, 2], cross_project[:, 2], project_specific[:, 2])).T

# Perform Nemenyi’s post-hoc test
nemenyi_precision = posthoc_nemenyi_friedman(data_precision)
nemenyi_recall = posthoc_nemenyi_friedman(data_recall)
nemenyi_f1 = posthoc_nemenyi_friedman(data_f1)

# Print results
print("=" * 50)
print("Friedman Test Results for Requirement Extraction")
print("=" * 50)
print(f"Precision: Chi-square = {friedman_precision.statistic:.3f}, p-value = {friedman_precision.pvalue:.3f}")
print(f"Recall: Chi-square = {friedman_recall.statistic:.3f}, p-value = {friedman_recall.pvalue:.3f}")
print(f"F1-score: Chi-square = {friedman_f1.statistic:.3f}, p-value = {friedman_f1.pvalue:.3f}")
print("=" * 50)

if friedman_precision.pvalue < 0.05 or friedman_recall.pvalue < 0.05 or friedman_f1.pvalue < 0.05:
    print("At least one metric shows significant differences between prompting strategies. Performing Nemenyi's post-hoc test...")

    # Print Nemenyi test results
    print("\nNemenyi's Post-hoc Test for Precision:")
    print(nemenyi_precision)

    print("\nNemenyi's Post-hoc Test for Recall:")
    print(nemenyi_recall)

    print("\nNemenyi's Post-hoc Test for F1-score:")
    print(nemenyi_f1)

    print("\nInterpretation:")
    print("If a p-value < 0.05 in the Nemenyi test, the corresponding prompting strategies are significantly different.")

else:
    print("No statistically significant differences detected between prompting strategies.")
