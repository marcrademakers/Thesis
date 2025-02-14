import os
import json
from collections import defaultdict

def load_json(file_path):
    """
    Load JSON data from a file.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def compare_tagged_data(original_file, new_file):
    """
    Compare two JSON files and compute differences.
    """
    original_data = load_json(original_file)
    new_data = load_json(new_file)

    stats = {
        'total_projects_original': len(original_data),
        'total_projects_new': len(new_data),
        'projects_with_differences': 0,
        'total_references_original': 0,
        'total_references_new': 0,
        'differences': defaultdict(dict),
    }

    for project, original_references in original_data.items():
        stats['total_references_original'] += len(original_references)

        if project in new_data:
            new_references = new_data[project]
            stats['total_references_new'] += len(new_references)

            if set(original_references) != set(new_references):
                stats['projects_with_differences'] += 1
                stats['differences'][project] = {
                    'only_in_original': list(set(original_references) - set(new_references)),
                    'only_in_new': list(set(new_references) - set(original_references)),
                }
        else:
            stats['projects_with_differences'] += 1
            stats['differences'][project] = {
                'only_in_original': original_references,
                'only_in_new': [],
            }

    for project, new_references in new_data.items():
        if project not in original_data:
            stats['total_references_new'] += len(new_references)
            stats['projects_with_differences'] += 1
            stats['differences'][project] = {
                'only_in_original': [],
                'only_in_new': new_references,
            }

    return stats

def compare_all_files(file_pairs, output_file):
    """
    Compare all corresponding JSON files in the provided file pairs.
    """
    results = {}

    for original_file, new_file in file_pairs:
        try:
            print(f"Comparing {os.path.basename(original_file)} with {os.path.basename(new_file)}...")
            results[os.path.basename(original_file)] = compare_tagged_data(original_file, new_file)
        except Exception as e:
            print(f"Error comparing {original_file} and {new_file}: {e}")

    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(results, file, ensure_ascii=False, indent=4)
    print(f"Comparison results saved to {output_file}")

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
        ("/Users/marcrademakers/Desktop/Tagged data/low_nfr.json", "/Users/marcrademakers/Desktop/Tagged data 2/low_nfr_2.json"),
    ]

    output_file = "/Users/marcrademakers/Desktop/RQ2/comparison_results.json"
    compare_all_files(file_pairs, output_file)
