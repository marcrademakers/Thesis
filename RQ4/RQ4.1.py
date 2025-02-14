import json
import pandas as pd

# File paths
extracted_baselines_file = "/Users/marcrademakers/Desktop/RQ4/extracted_baselines.json"
ground_truth_file = "/Users/marcrademakers/Desktop/RQ4/rq4groundtruth.json"

# Load extracted baseline requirements
with open(extracted_baselines_file, 'r', encoding='utf-8') as f:
    extracted_baselines = json.load(f)

# Load ground truth categories
with open(ground_truth_file, 'r', encoding='utf-8') as f:
    ground_truth = json.load(f)

# 🔎 **Step 1: Define category groups more flexibly**
category_groups = {
    "nfr": [],
    "user": [],
    "system": [],
    "high": [],
    "medium": [],
    "low": []
}

# Assign each requirement to multiple categories (type & level)
requirement_to_categories = {}
category_counts = {key: 0 for key in category_groups.keys()}  # Track total ground truth requirements per category

for category, projects in ground_truth.items():
    for project_name, requirements in projects.items():
        for requirement in requirements:
            # Normalize requirement: Lowercase, strip spaces, and remove trailing punctuation
            normalized_requirement = requirement.strip().lower().rstrip(".!?")

            # Determine both the type (NFR/User/System) and level (High/Medium/Low)
            assigned_types = []
            assigned_levels = []

            if "nfr" in category.lower():
                assigned_types.append("nfr")
            if "user" in category.lower():
                assigned_types.append("user")
            if "system" in category.lower():
                assigned_types.append("system")

            if "high" in category.lower():
                assigned_levels.append("high")
            if "medium" in category.lower():
                assigned_levels.append("medium")
            if "low" in category.lower():
                assigned_levels.append("low")

            # Ensure that every requirement is assigned to at least one type and one level
            for type_key in assigned_types:
                requirement_to_categories.setdefault(normalized_requirement, []).append(type_key)
                category_counts[type_key] += 1

            for level_key in assigned_levels:
                requirement_to_categories.setdefault(normalized_requirement, []).append(level_key)
                category_counts[level_key] += 1

# 🔎 **Step 2: Compare extracted baselines against ground truth**
correct_counts = {setup: {key: 0 for key in category_groups.keys()} for setup in extracted_baselines}

for setup, extracted_requirements in extracted_baselines.items():
    for requirement in extracted_requirements:
        # Normalize extracted requirement
        normalized_requirement = requirement.strip().lower().rstrip(".!?")

        # Track TP if requirement exists in ground truth
        if normalized_requirement in requirement_to_categories:
            for category in requirement_to_categories[normalized_requirement]:
                correct_counts[setup][category] += 1  # Count correct matches (TP)

# 🔎 **Step 3: Compute Recall per Category (Now Between 0 and 1)**
recall_results = {
    setup: {
        category: (correct_counts[setup][category] / category_counts[category]) if category_counts[category] > 0 else 0
        for category in category_groups.keys()
    }
    for setup in extracted_baselines
}

# Convert results to DataFrame
df_results = pd.DataFrame(recall_results)

# 🔎 **Step 4: Split into Two 3x3 Tables**
df_type = df_results.loc[["nfr", "user", "system"]]
df_level = df_results.loc[["high", "medium", "low"]]

# Print results in terminal in a clean format
print("\nRecall by Requirement Type (NFR, User, System):\n")
print(df_type)

print("\nRecall by Requirement Level (High, Medium, Low):\n")
print(df_level)
