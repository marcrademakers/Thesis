import json
import os

# File paths
comparison_files = {
    "Setup0": "/Users/marcrademakers/Desktop/Setup0/ID/comparison01.json",
    "Setup1": "/Users/marcrademakers/Desktop/Setup1/ID/comparison11.json",
    "Setup2": "/Users/marcrademakers/Desktop/Setup2/ID/comparison21.json",
}

# Similarity threshold for extraction
SIMILARITY_THRESHOLD = 0.7

# Output file path
output_file = "/Users/marcrademakers/Desktop/RQ4/extracted_baselines.json"

# Dictionary to store extracted baseline requirements
extracted_baselines = {setup: [] for setup in comparison_files}

# Iterate over each comparison file
for setup_name, file_path in comparison_files.items():
    with open(file_path, 'r', encoding='utf-8') as f:
        predicted_data = json.load(f)

    # Iterate through each project in the predicted data
    for project_data in predicted_data.values():
        for match in project_data.get("matches", []):
            # Check if the Best Similarity Score meets the threshold
            if match["Best Similarity Score"] >= SIMILARITY_THRESHOLD:
                extracted_baselines[setup_name].append(match["Baseline Requirement"])

# Save extracted baselines to a JSON file
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(extracted_baselines, f, indent=4, ensure_ascii=False)

print(f"✅ Extracted baselines saved to {output_file}")
