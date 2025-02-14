import os
import json
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score, confusion_matrix

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

# Set base directories
base_dir_1 = "/Users/marcrademakers/Desktop/Tagged data"
base_dir_2 = "/Users/marcrademakers/Desktop/Tagged data 2"

def load_json(file_path):
    """Load JSON file and return a set of requirements."""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return set()
    
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {req for sublist in data.values() for req in sublist}

# Step 1: Collect ALL unique references from all files
all_references = set()
round_1_references = set()
round_2_references = set()

for file1, file2 in file_pairs:
    set1 = load_json(os.path.join(base_dir_1, file1))
    set2 = load_json(os.path.join(base_dir_2, file2))
    
    round_1_references.update(set1)
    round_2_references.update(set2)
    all_references.update(set1)
    all_references.update(set2)

# Step 2: Assign binary labels for presence
round_1_presence = [1 if ref in round_1_references else 0 for ref in all_references]
round_2_presence = [1 if ref in round_2_references else 0 for ref in all_references]

# Step 3: Compute total reference counts
total_references = len(all_references)  # Total unique references across both rounds
total_round_1 = len(round_1_references)  # References in Round 1
total_round_2 = len(round_2_references)  # References in Round 2

# Step 4: Compute Cohen's Kappa
cohen_kappa = cohen_kappa_score(round_1_presence, round_2_presence)

# Step 5: Compute agreement and disagreement counts
agreement_count = sum(1 for a, b in zip(round_1_presence, round_2_presence) if a == b)
disagreement_count = total_references - agreement_count

# Step 6: Compute confusion matrix (showing disagreements correctly)
cm = confusion_matrix(round_1_presence, round_2_presence, labels=[0, 1])

# Extract values from the confusion matrix
tn, fp, fn, tp = cm.ravel()  # True Negatives, False Positives, False Negatives, True Positives

# Step 7: Print results
print("\nCorrected Confusion Matrix:")
print(f"{'':<12}{'Round 2: 0':<12}{'Round 2: 1':<12}")
print(f"Round 1: 0  {tn:<12}{fp:<12}")
print(f"Round 1: 1  {fn:<12}{tp:<12}")

print(f"\nCohen's Kappa for tagging agreement: {cohen_kappa:.4f}")
print(f"Total References in Round 1: {total_round_1} (should match summary script)")
print(f"Total References in Round 2: {total_round_2} (should match summary script)")
print(f"Total Unique References Considered: {total_references} (should be 1550)")
print(f"Total Agreements: {agreement_count}")
print(f"Total Disagreements: {disagreement_count}")
print(f"True Negatives (Not Labeled in Both Rounds): {tn}")
print(f"False Positives (New in Round 2): {fp}")
print(f"False Negatives (Removed from Round 1): {fn}")
print(f"True Positives (Labeled in Both Rounds): {tp}")

# Step 8: Visualization
plt.figure(figsize=(6, 4))
plt.bar(["Agreements", "Disagreements"], [agreement_count, disagreement_count], color=["blue", "red"])
plt.title("Agreement vs Disagreement in Requirement Tagging")
plt.ylabel("Count")
plt.show()
