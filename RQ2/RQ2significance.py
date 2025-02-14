import numpy as np
from scipy.stats import chi2_contingency, ttest_rel, wilcoxon, binomtest
from statsmodels.stats.proportion import proportions_ztest

# Data for all categories
categories = {
    "high_nfr": {"refs_1": 40, "refs_2": 24, "unique_1": 19, "unique_2": 3, "exact_matches": 21},
    "high_system": {"refs_1": 9, "refs_2": 8, "unique_1": 2, "unique_2": 1, "exact_matches": 7},
    "high_user": {"refs_1": 41, "refs_2": 46, "unique_1": 7, "unique_2": 12, "exact_matches": 33},
    "medium_user": {"refs_1": 430, "refs_2": 499, "unique_1": 69, "unique_2": 135, "exact_matches": 358},
    "medium_system": {"refs_1": 115, "refs_2": 104, "unique_1": 57, "unique_2": 46, "exact_matches": 57},
    "medium_nfr": {"refs_1": 318, "refs_2": 178, "unique_1": 178, "unique_2": 40, "exact_matches": 137},
    "low_user": {"refs_1": 159, "refs_2": 180, "unique_1": 27, "unique_2": 48, "exact_matches": 132},
    "low_system": {"refs_1": 76, "refs_2": 65, "unique_1": 30, "unique_2": 18, "exact_matches": 46},
    "low_nfr": {"refs_1": 49, "refs_2": 50, "unique_1": 24, "unique_2": 24, "exact_matches": 25},
}

# Prepare data for overall statistical tests
unique_1 = [data["unique_1"] for data in categories.values()]
unique_2 = [data["unique_2"] for data in categories.values()]
refs_1 = [data["refs_1"] for data in categories.values()]
refs_2 = [data["refs_2"] for data in categories.values()]

# Chi-Square Test for distribution differences
contingency_table = np.array([unique_1, unique_2])
chi2, p_chi2, _, _ = chi2_contingency(contingency_table)
print(f"Chi-Square Test: chi2 = {chi2:.3f}, p-value = {p_chi2:.5f}")

# Paired t-Test for total references
t_stat, p_ttest = ttest_rel(refs_1, refs_2)
print(f"Paired t-Test: t = {t_stat:.3f}, p-value = {p_ttest:.5f}")

# Wilcoxon Signed-Rank Test for total references
try:
    w_stat, p_wilcoxon = wilcoxon(refs_1, refs_2)
    print(f"Wilcoxon Signed-Rank Test: W = {w_stat:.3f}, p-value = {p_wilcoxon:.5f}")
except ValueError as e:
    print(f"Wilcoxon Signed-Rank Test could not be performed: {e}")

# Analyze specific categories with binomial and proportion tests
print("\nCategory-Specific Tests:")
for category, data in categories.items():
    total_unique = data["unique_1"] + data["unique_2"]
    print(f"\nCategory: {category}")
    print(f"  Unique References (Round 1): {data['unique_1']}")
    print(f"  Unique References (Round 2): {data['unique_2']}")
    print(f"  Exact Matches: {data['exact_matches']}")

    # Binomial Test
    p_binom = binomtest(data["unique_1"], n=total_unique, p=0.5, alternative="two-sided").pvalue
    print(f"  Binomial Test: p-value = {p_binom:.5f}")

    # Proportion Z-Test
    counts = [data["unique_1"], data["unique_2"]]
    nobs = [total_unique, total_unique]
    z_stat, p_ztest = proportions_ztest(counts, nobs, alternative="two-sided")
    print(f"  Proportion Z-Test: Z = {z_stat:.5f}, p-value = {p_ztest:.5f}")
