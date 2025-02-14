import os
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import cohen_kappa_score

def load_json(file_path):
    """
    Load JSON data from a file.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def classify_category(category):
    """
    Classify categories into granularity levels and types.
    """
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
    Compute two matrices:
    - One for granularity shifts
    - One for type shifts
    - Track removed requirements
    """
    granularity_matrix = defaultdict(lambda: defaultdict(int))
    type_matrix = defaultdict(lambda: defaultdict(int))
    reference_to_category = {}
    labels_original_granularity = []
    labels_new_granularity = []
    labels_original_type = []
    labels_new_type = []
    labels_original_granularity_extended = []
    labels_new_granularity_extended = []
    labels_original_type_extended = []
    labels_new_type_extended = []
    removed_references = defaultdict(int)
    
    # Collect all references and their categories from round 1
    for original_file, _ in file_pairs:
        original_data = load_json(original_file)
        category = os.path.basename(original_file).replace('.json', '')
        granularity, req_type = classify_category(category)
        
        for project, references in original_data.items():
            for ref in references:
                reference_to_category[ref] = (granularity, req_type)
    
    # Track where they appear in round 2
    found_references = set()
    for _, new_file in file_pairs:
        new_data = load_json(new_file)
        category = os.path.basename(new_file).replace('.json', '')
        new_granularity, new_type = classify_category(category)
        
        for project, references in new_data.items():
            for ref in references:
                found_references.add(ref)
                old_granularity, old_type = reference_to_category.get(ref, ('new', 'new'))
                granularity_matrix[old_granularity][new_granularity] += 1
                type_matrix[old_type][new_type] += 1
                labels_original_granularity.append(old_granularity)
                labels_new_granularity.append(new_granularity)
                labels_original_type.append(old_type)
                labels_new_type.append(new_type)
                labels_original_granularity_extended.append(old_granularity)
                labels_new_granularity_extended.append(new_granularity)
                labels_original_type_extended.append(old_type)
                labels_new_type_extended.append(new_type)
    
    # Identify removed references
    for ref, (old_granularity, old_type) in reference_to_category.items():
        if ref not in found_references:
            granularity_matrix[old_granularity]["removed"] += 1
            type_matrix[old_type]["removed"] += 1
            removed_references[old_granularity] += 1
            labels_original_granularity_extended.append(old_granularity)
            labels_new_granularity_extended.append("removed")
            labels_original_type_extended.append(old_type)
            labels_new_type_extended.append("removed")
    
    all_granularities = ["high", "medium", "low"]
    all_types = ["nfr", "system", "user"]
    
    all_granularities_extended = ["high", "medium", "low", "new", "removed"]
    all_types_extended = ["nfr", "system", "user", "new", "removed"]
    
    cohen_kappa_granularity = cohen_kappa_score(
        labels_original_granularity, labels_new_granularity, labels=all_granularities
    ) if len(set(labels_original_granularity)) > 1 and len(set(labels_new_granularity)) > 1 else float('nan')
    
    cohen_kappa_type = cohen_kappa_score(
        labels_original_type, labels_new_type, labels=all_types
    ) if len(set(labels_original_type)) > 1 and len(set(labels_new_type)) > 1 else float('nan')
    
    cohen_kappa_granularity_extended = cohen_kappa_score(
        labels_original_granularity_extended, labels_new_granularity_extended, labels=all_granularities_extended
    ) if len(set(labels_original_granularity_extended)) > 1 and len(set(labels_new_granularity_extended)) > 1 else float('nan')
    
    cohen_kappa_type_extended = cohen_kappa_score(
        labels_original_type_extended, labels_new_type_extended, labels=all_types_extended
    ) if len(set(labels_original_type_extended)) > 1 and len(set(labels_new_type_extended)) > 1 else float('nan')

    return (
        granularity_matrix, type_matrix, 
        cohen_kappa_granularity, cohen_kappa_type, 
        cohen_kappa_granularity_extended, cohen_kappa_type_extended, 
        removed_references
    )

def save_results(output_file, granularity_matrix, type_matrix, cohen_kappa_granularity, cohen_kappa_type, cohen_kappa_granularity_extended, cohen_kappa_type_extended, removed_references):
    """
    Save results to a structured JSON file.
    """
    results = {
        'granularity_matrix': granularity_matrix,
        'type_matrix': type_matrix,
        'cohen_kappa_granularity': cohen_kappa_granularity,
        'cohen_kappa_type': cohen_kappa_type,
        'cohen_kappa_granularity_extended': cohen_kappa_granularity_extended,
        'cohen_kappa_type_extended': cohen_kappa_type_extended,
        'removed_references': removed_references
    }
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(results, file, ensure_ascii=False, indent=4)
    print(f"Aggregated results saved to {output_file}")

def compare_all_files(file_pairs, output_file):
    """
    Compare all file pairs and generate statistics.
    """
    (
        granularity_matrix, type_matrix, 
        cohen_kappa_granularity, cohen_kappa_type, 
        cohen_kappa_granularity_extended, cohen_kappa_type_extended, 
        removed_references
    ) = compute_shift_matrices(file_pairs)

    save_results(
        output_file, granularity_matrix, type_matrix, 
        cohen_kappa_granularity, cohen_kappa_type, 
        cohen_kappa_granularity_extended, cohen_kappa_type_extended, 
        removed_references
    )

# Main Script
# Main Script
if __name__ == "__main__":
    file_pairs = [
        ("/Users/marcrademakers/Desktop/Tagged data/high_nfr.json", "/Users/marcrademakers/Desktop/Tagged data 2/high_nfr_2.json"),
        ("/Users/marcrademakers/Desktop/Tagged data/high_system.json", "/Users/marcrademakers/Desktop/Tagged data 2/high_system_2.json"),
        ("/Users/marcrademakers/Desktop/Tagged data/high_user.json", "/Users/marcrademakers/Desktop/Tagged data 2/high_user_2.json"),
        ("/Users/marcrademakers/Desktop/Tagged data/medium_user.json", "/Users/marcrademakers/Desktop/Tagged data 2/medium_user_2.json"),
        ("/Users/marcrademakers/Desktop/Tagged data/medium_system.json", "/Users/marcrademakers/Desktop/Tagged data 2/medium_system_2.json"),
        ("/Users/marcrademakers/Desktop/Tagged data/medium_nfr.json", "/Users/marcrademakers/Desktop/Tagged data 2/medium_nfr_2.json"),
        ("/Users/marcrademakers/Desktop/Tagged data/low_user.json", "/Users/marcrademakers/Desktop/Tagged data 2/low_user_2.json"),
        ("/Users/marcrademakers/Desktop/Tagged data/low_system.json", "/Users/marcrademakers/Desktop/Tagged data 2/low_system_2.json"),
        ("/Users/marcrademakers/Desktop/Tagged data/low_nfr.json", "/Users/marcrademakers/Desktop/Tagged data 2/low_nfr_2.json")
    ]
    
    output_file = "/Users/marcrademakers/Desktop/matrix.json"
    
    compare_all_files(file_pairs, output_file)
