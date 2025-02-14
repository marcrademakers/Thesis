import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import cohen_kappa_score

def load_json(file_path):
    """Load JSON data from a file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def classify_category(category):
    """Classify categories into granularity levels and types."""
    if "high" in category:
        granularity = "high"
    elif "medium" in category:
        granularity = "medium"
    else:
        granularity = "low"
    
    if "nfr" in category:
        req_type = "nfr"
    elif "system" in category:
        req_type = "system"
    else:
        req_type = "user"
    
    return granularity, req_type

def compute_shift_matrices(file_pairs):
    """
    Compute four 3x3 matrices:
    1. Granularity shifts (high, medium, low)
    2. Type shifts (nfr, system, user)
    3. New references (appeared in round 2 but not in round 1)
    4. Removed references (were in round 1 but missing in round 2)
    """
    granularity_matrix = defaultdict(lambda: defaultdict(int))
    type_matrix = defaultdict(lambda: defaultdict(int))
    new_entries_matrix = defaultdict(lambda: defaultdict(int))
    removed_entries_matrix = defaultdict(lambda: defaultdict(int))
    
    reference_to_category = {}
    found_references = set()
    round_1_references = set()
    round_2_references = set()

    # Lists for Cohen’s Kappa computation
    labels_original_granularity = []
    labels_new_granularity = []
    labels_original_type = []
    labels_new_type = []

    # Collect all references from round 1
    for original_file, _ in file_pairs:
        original_data = load_json(original_file)
        category = os.path.basename(original_file).replace('.json', '')
        granularity, req_type = classify_category(category)
        
        for project, references in original_data.items():
            for ref in references:
                reference_to_category[ref] = (granularity, req_type)
                round_1_references.add(ref)

    # Track references in round 2
    for _, new_file in file_pairs:
        new_data = load_json(new_file)
        category = os.path.basename(new_file).replace('.json', '')
        new_granularity, new_type = classify_category(category)
        
        for project, references in new_data.items():
            for ref in references:
                found_references.add(ref)
                round_2_references.add(ref)

                if ref in reference_to_category:
                    # Reference existed in round 1, track shifts
                    old_granularity, old_type = reference_to_category[ref]
                    granularity_matrix[old_granularity][new_granularity] += 1
                    type_matrix[old_type][new_type] += 1

                    # Store labels for Cohen’s Kappa calculation
                    labels_original_granularity.append(old_granularity)
                    labels_new_granularity.append(new_granularity)
                    labels_original_type.append(old_type)
                    labels_new_type.append(new_type)
                else:
                    # Reference is new in round 2
                    new_entries_matrix[new_granularity][new_type] += 1

    # Identify removed references
    for ref, (old_granularity, old_type) in reference_to_category.items():
        if ref not in found_references:
            removed_entries_matrix[old_granularity][old_type] += 1

    # Compute Cohen's Kappa for Granularity and Type Shifts
    all_granularities = ["high", "medium", "low"]
    all_types = ["nfr", "system", "user"]

    cohen_kappa_granularity = cohen_kappa_score(
        labels_original_granularity, labels_new_granularity, labels=all_granularities
    ) if len(set(labels_original_granularity)) > 1 and len(set(labels_new_granularity)) > 1 else float('nan')

    cohen_kappa_type = cohen_kappa_score(
        labels_original_type, labels_new_type, labels=all_types
    ) if len(set(labels_original_type)) > 1 and len(set(labels_new_type)) > 1 else float('nan')

    return (
        granularity_matrix, type_matrix, 
        new_entries_matrix, removed_entries_matrix,
        cohen_kappa_granularity, cohen_kappa_type,
        round_1_references, round_2_references
    )

def print_summary_stats(round_1_references, round_2_references, cohen_kappa_granularity, cohen_kappa_type):
    """
    Print a summary of reference counts, agreement, and shifts.
    """
    total_ref_1 = len(round_1_references)
    total_ref_2 = len(round_2_references)
    
    exact_matches = len(round_1_references & round_2_references)
    removed = len(round_1_references - round_2_references)
    new = len(round_2_references - round_1_references)
    total_unique_refs = len(round_1_references | round_2_references)
    agreement_percentage = (exact_matches / total_unique_refs) * 100

    print("\n=== Summary Statistics ===")
    print(f"Total References in Round 1: {total_ref_1}")
    print(f"Total References in Round 2: {total_ref_2}")
    print(f"Total Unique References Considered: {total_unique_refs}")
    print(f"Exact Matches: {exact_matches}")
    print(f"New References in Round 2: {new}")
    print(f"Removed References in Round 1: {removed}")
    print(f"Overall Agreement: {agreement_percentage:.2f}%")
    print(f"Cohen's Kappa for Granularity Shifts: {cohen_kappa_granularity:.4f}")
    print(f"Cohen's Kappa for Type Shifts: {cohen_kappa_type:.4f}")

def compare_all_files(file_pairs):
    """
    Compare all file pairs and generate statistics.
    """
    (
        granularity_matrix, type_matrix, 
        new_entries_matrix, removed_entries_matrix,
        cohen_kappa_granularity, cohen_kappa_type,
        round_1_references, round_2_references
    ) = compute_shift_matrices(file_pairs)

    print_summary_stats(round_1_references, round_2_references, cohen_kappa_granularity, cohen_kappa_type)

    # Convert to DataFrame for heatmap plotting
    df_granularity = pd.DataFrame(granularity_matrix).fillna(0).astype(int)
    df_type = pd.DataFrame(type_matrix).fillna(0).astype(int)

    # Generate heatmaps
    plot_heatmap(df_granularity, "Granularity Shift Matrix")
    plot_heatmap(df_type, "Type Shift Matrix", reorder_axes=True)

def plot_heatmap(matrix, title, reorder_axes=False):
    """
    Plot a heatmap for the given matrix.
    If `reorder_axes` is True, the type matrix is reordered as required.
    """
    plt.figure(figsize=(6, 5))

    if reorder_axes:
        order_y = ["user", "system", "nfr"]
        order_x = ["nfr", "system", "user"]
        matrix = matrix.reindex(index=order_y, columns=order_x, fill_value=0)

    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", linewidths=0.5)
    plt.title(title)
    plt.xlabel("Round 2")
    plt.ylabel("Round 1")
    plt.show()

# Main Script
if __name__ == "__main__":
    file_pairs = [(f"/Users/marcrademakers/Desktop/Tagged data/{file1}", 
                   f"/Users/marcrademakers/Desktop/Tagged data 2/{file2}") 
                  for file1, file2 in [
                      ("high_nfr.json", "high_nfr_2.json"),
                      ("high_system.json", "high_system_2.json"),
                      ("high_user.json", "high_user_2.json"),
                      ("medium_user.json", "medium_user_2.json"),
                      ("medium_system.json", "medium_system_2.json"),
                      ("medium_nfr.json", "medium_nfr_2.json"),
                      ("low_user.json", "low_user_2.json"),
                      ("low_system.json", "low_system_2.json"),
                      ("low_nfr.json", "low_nfr_2.json")
                  ]]

    compare_all_files(file_pairs)
