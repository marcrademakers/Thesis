import os
import json
import numpy as np
import krippendorff
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

def compute_general_statistics(original_data, new_data):
    """
    Compute general statistics between two datasets.
    """
    return {
        'total_projects_original': len(original_data),
        'total_projects_new': len(new_data),
        'projects_in_both': len(set(original_data.keys()) & set(new_data.keys())),
        'total_references_original': sum(len(v) for v in original_data.values()),
        'total_references_new': sum(len(v) for v in new_data.values())
    }

def compute_transition_matrix(file_pairs):
    """
    Compute a transition matrix across all file pairs to track category shifts.
    """
    transition_matrix = defaultdict(lambda: defaultdict(int))
    
    reference_to_category = {}
    
    # First, collect all references and their categories from round 1
    for original_file, _ in file_pairs:
        original_data = load_json(original_file)
        category = os.path.basename(original_file).replace('.json', '')
        
        for project, references in original_data.items():
            for ref in references:
                reference_to_category[ref] = category
    
    # Now, track where they appear in round 2
    for _, new_file in file_pairs:
        new_data = load_json(new_file)
        new_category = os.path.basename(new_file).replace('.json', '')
        
        for project, references in new_data.items():
            for ref in references:
                old_category = reference_to_category.get(ref, 'new')
                transition_matrix[old_category][new_category] += 1
    
    return transition_matrix

def calculate_cohen_kappa(original_data, new_data):
    """
    Calculate Cohen's Kappa for inter-rater agreement.
    """
    all_references = list(set(original_data.keys()) | set(new_data.keys()))
    
    labels_original = [1 if ref in original_data else 0 for ref in all_references]
    labels_new = [1 if ref in new_data else 0 for ref in all_references]
    
    return cohen_kappa_score(labels_original, labels_new)

def plot_transition_heatmap(transition_matrix, output_file):
    """
    Generate a heatmap for category shifts.
    """
    categories = sorted(set(transition_matrix.keys()) | {cat for sub in transition_matrix.values() for cat in sub.keys()})
    matrix = np.zeros((len(categories), len(categories)))
    
    for i, from_cat in enumerate(categories):
        for j, to_cat in enumerate(categories):
            matrix[i, j] = transition_matrix.get(from_cat, {}).get(to_cat, 0)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt="g", cmap="Blues", xticklabels=categories, yticklabels=categories)
    plt.xlabel("New Category")
    plt.ylabel("Original Category")
    plt.title("Requirement Category Transition Matrix")
    plt.savefig(output_file)
    plt.close()

def save_results(output_file, aggregated_results):
    """
    Save results to a structured JSON file.
    """
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(aggregated_results, file, ensure_ascii=False, indent=4)
    print(f"Aggregated results saved to {output_file}")

def compare_all_files(file_pairs, output_file, heatmap_file):
    """
    Compare all file pairs and generate statistics.
    """
    transition_matrix = compute_transition_matrix(file_pairs)
    
    save_results(output_file, {'transition_matrix': transition_matrix})
    plot_transition_heatmap(transition_matrix, heatmap_file)

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
    
    output_file = "/Users/marcrademakers/Desktop/rq2_updated.json"
    heatmap_file = "/Users/marcrademakers/Desktop/rq2_heatmap.png"
    compare_all_files(file_pairs, output_file, heatmap_file)
