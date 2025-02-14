import json
import os

# Define file pairs
file_pairs = [
    ("high_nfr.json", "high_nfr_2.json"),
    ("high_system.json", "high_system_2.json"),
    ("high_user.json", "high_user_2.json"),
    ("medium_user.json", "medium_user_2.json"),
    ("medium_system.json", "medium_system_2.json"),
    ("medium_nfr.json", "medium_nfr_2.json"),
    ("low_user.json", "low_user_2.json"),
    ("low_system.json", "low_system_2.json"),
    ("low_nfr.json", "low_nfr_2.json"),
]

# Set base directories for the JSON files
base_dir_1 = "/Users/marcrademakers/Desktop/Tagged data"
base_dir_2 = "/Users/marcrademakers/Desktop/Tagged data 2"

def load_json(file_path):
    """Load JSON file and return a set of unique requirements."""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return set()
    
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Convert to a set of unique references
    return {req for sublist in data.values() for req in sublist}

# Global tracking of unique references to prevent double counting
global_seen_ref_1 = set()
global_seen_ref_2 = set()
global_all_references = set()  # Used to ensure proper unique count

# Print header
print(f"{'Category':<20}{'Ref 1':<8}{'Ref 2':<8}{'Unique 1':<10}{'Unique 2':<10}{'Matches':<10}{'Agreement (%)':<15}")
print("=" * 80)

# Initialize totals
total_ref_1 = 0
total_ref_2 = 0
total_unique_1 = 0
total_unique_2 = 0
total_matches = 0
total_comparisons = 0

# Process each file pair
for file1, file2 in file_pairs:
    path1 = os.path.join(base_dir_1, file1)
    path2 = os.path.join(base_dir_2, file2)

    category = file1.replace(".json", "").replace("_", " ")

    # Load data
    set1 = load_json(path1)
    set2 = load_json(path2)

    # Get references that haven't been counted in previous categories
    new_set1 = set1 - global_seen_ref_1
    new_set2 = set2 - global_seen_ref_2

    # Update global tracking for references
    global_seen_ref_1.update(set1)
    global_seen_ref_2.update(set2)
    global_all_references.update(set1 | set2)  # Track all seen references

    # Compute unique references per round
    unique_1 = len(new_set1 - new_set2)  # Unique in round 1
    unique_2 = len(new_set2 - new_set1)  # Unique in round 2
    exact_matches = len(new_set1 & new_set2)  # Matching references

    # Agreement Calculation
    total_compare = exact_matches + unique_1 + unique_2
    agreement = (exact_matches / total_compare * 100) if total_compare > 0 else 0

    # Update totals
    total_ref_1 += len(new_set1)
    total_ref_2 += len(new_set2)
    total_unique_1 += unique_1
    total_unique_2 += unique_2
    total_matches += exact_matches
    total_comparisons += total_compare

    # Print per-category counts
    print(f"{category:<20}{len(new_set1):<8}{len(new_set2):<8}{unique_1:<10}{unique_2:<10}{exact_matches:<10}{agreement:<15.2f}")

# Compute overall agreement percentage
overall_agreement = (total_matches / total_comparisons * 100) if total_comparisons > 0 else 0

# Print totals row
print("=" * 80)
print(f"{'TOTAL (Category-Based)':<20}{total_ref_1:<8}{total_ref_2:<8}{total_unique_1:<10}{total_unique_2:<10}{total_matches:<10}{overall_agreement:<15.2f}")

# Compute final deduplicated unique references
total_unique_references = len(global_all_references)

# Print final deduplicated counts
print("=" * 80)
print(f"{'TOTAL Unique References':<20}{total_unique_references:<8}")
