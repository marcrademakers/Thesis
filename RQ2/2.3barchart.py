import json
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, cohen_kappa_score

# Paths to merged files
merged_file_1 = "/Users/marcrademakers/Desktop/merged_data_1.json"
merged_file_2 = "/Users/marcrademakers/Desktop/merged_data_2.json"

# Mapping original JSON keys to preferred project names
project_name_mapping = {
    "\\\\Cost_Management": "Cost Management",
    "\\\\Jira_Performance_Testing_Tools": "Jira Performance Testing Tools",
    "\\\\Lyrasis Dura Cloud": "Lyrasis Dura Cloud",
    "\\\\Network_Observability": "Network Observability",
    "\\\\OpenShift_UX_Product_Design": "OpenShift UX Product Design",
    "\\\\Qt_Design_Studio": "Qt Design Studio",
    "\\\\Red_Hat_Developer_Website_v2": "Red Hat Developer Website",
}

# Define the correct project display order
project_order = [
    "Cost Management",
    "Jira Performance Testing Tools",
    "Lyrasis Dura Cloud",
    "Network Observability",
    "OpenShift UX Product Design",
    "Qt Design Studio",
    "Red Hat Developer Website",
]

# Load JSON files
def load_json(file_path):
    """Load JSON file and return its data as a dictionary."""
    if not os.path.exists(file_path):
        print(f"Error: File not found - {file_path}")
        return {}
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# Load merged data
data_1 = load_json(merged_file_1)
data_2 = load_json(merged_file_2)

# Collect agreement and disagreement counts for each project
project_agreements = []
project_disagreements = []

# Collect overall statistics
all_references = set()
round_1_references = set()
round_2_references = set()

# Print per-project statistics
print("\nProject-wise Agreement and Disagreement Counts:")
print("-" * 50)

# Compare projects in the specified order
for json_key, display_name in project_name_mapping.items():
    set1 = set(data_1.get(json_key, []))  # Defaults to empty set if missing
    set2 = set(data_2.get(json_key, []))

    round_1_references.update(set1)
    round_2_references.update(set2)
    all_references.update(set1)
    all_references.update(set2)

    agreement_count = len(set1.intersection(set2))
    disagreement_count = len(set1.symmetric_difference(set2))

    project_agreements.append(agreement_count)
    project_disagreements.append(disagreement_count)

    # Print individual project values
    print(f"{display_name}:")
    print(f"  - Agreements: {agreement_count}")
    print(f"  - Disagreements: {disagreement_count}")
    print("-" * 50)

# Compute total reference counts
total_references = len(all_references)  # Total unique references across both rounds
total_round_1 = len(round_1_references)  # References in Round 1
total_round_2 = len(round_2_references)  # References in Round 2

# Compute Cohen's Kappa
round_1_presence = [1 if ref in round_1_references else 0 for ref in all_references]
round_2_presence = [1 if ref in round_2_references else 0 for ref in all_references]
cohen_kappa = cohen_kappa_score(round_1_presence, round_2_presence)

# Compute agreement and disagreement counts
agreement_count = sum(1 for a, b in zip(round_1_presence, round_2_presence) if a == b)
disagreement_count = total_references - agreement_count

# Compute confusion matrix
cm = confusion_matrix(round_1_presence, round_2_presence, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()  # True Negatives, False Positives, False Negatives, True Positives

# Print Summary Statistics
print("\nComparison Summary:")
print(f"Total References in Round 1: {total_round_1}")
print(f"Total References in Round 2: {total_round_2}")
print(f"Total Unique References Considered: {total_references}")
print(f"Total Agreements: {agreement_count}")
print(f"Total Disagreements: {disagreement_count}")
print(f"True Negatives (Not Labeled in Both Rounds): {tn} (should be 0)")
print(f"False Positives (New in Round 2): {fp}")
print(f"False Negatives (Removed from Round 1): {fn}")
print(f"True Positives (Labeled in Both Rounds): {tp}")
print(f"\nCohen's Kappa Score: {cohen_kappa:.4f}")

# Create bar chart with fixed order
x = np.arange(len(project_order))
width = 0.4  

plt.figure(figsize=(12, 6))
plt.bar(x - width/2, project_agreements, width, label="Agreements", color="#6c8ebf")
plt.bar(x + width/2, project_disagreements, width, label="Disagreements", color="#d79b00")

plt.xlabel("Projects")
plt.ylabel("Count")
plt.title("Agreement vs Disagreement in Requirement Tagging by Project")
plt.xticks(ticks=x, labels=project_order, rotation=45, ha="right")
plt.legend()
plt.tight_layout()
plt.show()
