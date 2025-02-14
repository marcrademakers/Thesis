import json

# Load the JSON file
#input_file = "/Users/marcrademakers/Desktop/Setup0/ID/comparison01.json"
#input_file = "/Users/marcrademakers/Desktop/Setup1/ID/comparison11.json"
input_file = "/Users/marcrademakers/Desktop/Setup2/ID/comparison21.json"

with open(input_file, "r") as f:
    data = json.load(f)

# Compute total false positives across all projects
total_false_positives = sum(
    metrics.get("false_positives", 0) for project, metrics in data.items() if project != "summary"
)

# Print overall summary
print("\n=== Overall Summary ===")
print(f"Total Baselines: {data['summary']['total_baselines']}")
print(f"Total Automated Requirements: {data['summary']['total_automated_requirements']}")
print(f"Overall Precision: {data['summary']['overall_precision']:.4f}")
print(f"Overall Recall: {data['summary']['overall_recall']:.4f}")
print(f"Overall F1 Score: {data['summary']['overall_f1_score']:.4f}")
print(f"Total False Positives Across All Projects: {total_false_positives}\n")  # NEW LINE

# Print project-wise metrics
print("=== Project-Specific Metrics ===")
for project, metrics in data.items():
    if project == "summary":
        continue  # Skip the summary section

    print(f"\nProject: {project}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1 Score: {metrics['f1_score']:.4f}")
