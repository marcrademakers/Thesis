import numpy as np
import scipy.stats as stats

# Recall data for Requirement Type (NFR, User, System)
recall_type = np.array([
    [0.301653, 0.512397, 0.466942],  # Setup0
    [0.324713, 0.406609, 0.416667],  # Setup1
    [0.211180, 0.385093, 0.229814]   # Setup2
])

# Recall data for Requirement Level (High, Medium, Low)
recall_level = np.array([
    [0.041096, 0.315068, 0.191781],  # Setup0
    [0.300401, 0.500668, 0.409880],  # Setup1
    [0.379061, 0.256318, 0.429603]   # Setup2
])

# Perform Friedman Test for Requirement Type
friedman_type_stat, friedman_type_p = stats.friedmanchisquare(recall_type[:, 0], recall_type[:, 1], recall_type[:, 2])

# Perform Friedman Test for Requirement Level
friedman_level_stat, friedman_level_p = stats.friedmanchisquare(recall_level[:, 0], recall_level[:, 1], recall_level[:, 2])

# Print results
print("\n📊 Friedman Test Results for Requirement Type (NFR, User, System):")
print(f"Test Statistic: {friedman_type_stat:.4f}, p-value: {friedman_type_p:.4f}")

print("\n📊 Friedman Test Results for Requirement Level (High, Medium, Low):")
print(f"Test Statistic: {friedman_level_stat:.4f}, p-value: {friedman_level_p:.4f}")

# Interpretation
alpha = 0.05
if friedman_type_p < alpha:
    print("\n🔍 The differences in recall across requirement types are statistically significant (p < 0.05).")
else:
    print("\n🔍 No significant differences found across requirement types (p >= 0.05).")

if friedman_level_p < alpha:
    print("\n🔍 The differences in recall across requirement levels are statistically significant (p < 0.05).")
else:
    print("\n🔍 No significant differences found across requirement levels (p >= 0.05).")
