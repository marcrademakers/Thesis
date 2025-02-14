import json
import re
from collections import defaultdict
from torch.nn.functional import cosine_similarity
import torch
from sentence_transformers import SentenceTransformer

# Load the SBERT model
sbert_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

def normalize_text(text):
    """Normalizes text by stripping spaces and removing trailing punctuation."""
    text = text.strip()  # Remove leading/trailing spaces
    text = re.sub(r'[.,!?;:\-]+$', '', text)  # Remove punctuation at the end
    return text.lower()  # Convert to lowercase for consistency

def read_json_requirements(json_file):
    """Reads requirements from a JSON file."""
    with open(json_file, 'r') as file:
        data = json.load(file)
    return data  # Return the JSON structure directly

def filter_baseline_by_project(baseline_data, project_name):
    """Filters the baseline requirements for the specified project."""
    return baseline_data.get(project_name, [])

def get_similarity(text1, text2):
    """Computes semantic similarity between two pieces of text with vector normalization."""
    # Normalize text before encoding
    text1 = normalize_text(text1)
    text2 = normalize_text(text2)

    # Compute embeddings
    embedding1 = sbert_model.encode(text1, convert_to_tensor=True)
    embedding2 = sbert_model.encode(text2, convert_to_tensor=True)

    # Normalize embeddings to unit vectors
    embedding1 = embedding1 / embedding1.norm()
    embedding2 = embedding2 / embedding2.norm()

    # Compute cosine similarity
    similarity = cosine_similarity(embedding1.unsqueeze(0), embedding2.unsqueeze(0))
    return similarity.item()

def compare_files(baseline_file, automated_file, project_name, results_dict):
    """Compares baseline JSON requirements with automated requirements."""
    baseline_data = read_json_requirements(baseline_file)
    automated_contents = read_json_requirements(automated_file)
    
    # Filter baseline by project name (ground truth)
    baseline_requirements = filter_baseline_by_project(baseline_data, project_name)
    
    results = []
    unmatched_baselines = []
    unmatched_automated = set()
    threshold = 0.7  # Set meaningful similarity threshold
    
    # Convert automated requirements into a dictionary for quick lookup
    automated_dict = defaultdict(list)
    for item in automated_contents:
        automated_dict[item["id"]].append(normalize_text(item["requirement"]))  # Normalize here
        unmatched_automated.add(item["requirement"])
    
    total_baselines = len(baseline_requirements)
    correct_predictions = 0
    false_negatives = 0
    total_automated = len(automated_contents)

    # Compare baseline requirements to automated requirements
    for base_req in baseline_requirements:
        base_id = base_req["id"]
        base_text = normalize_text(base_req["requirement"])  # Normalize before comparison
        
        # Find matching automated requirements by ID
        auto_texts = automated_dict.get(base_id, [])
        
        if auto_texts:
            best_match = None
            best_similarity = 0
            all_matches = []
            
            for auto_text in auto_texts:
                similarity = get_similarity(base_text, auto_text)
                all_matches.append({"Requirement": auto_text, "Similarity": similarity})
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = auto_text
            
            if best_match:
                correct_predictions += 1 if best_similarity >= threshold else 0
                unmatched_automated.discard(best_match)
                results.append({
                    "Baseline Requirement": base_text,
                    "Baseline Requirement ID": base_id,
                    "Best Automated Requirement Match": best_match,
                    "Best Similarity Score": best_similarity,
                    "All Automated Matches": all_matches
                })
        else:
            false_negatives += 1
            unmatched_baselines.append({
                "Baseline Requirement": base_text,
                "Baseline Requirement ID": base_id
            })
    
    false_positives = total_automated - correct_predictions

    # Compute evaluation metrics
    precision = correct_predictions / (correct_predictions + false_positives) if (correct_predictions + false_positives) > 0 else 0
    recall = correct_predictions / (correct_predictions + false_negatives) if (correct_predictions + false_negatives) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Store results for project
    results_dict[project_name] = {
        "total_baselines": total_baselines,
        "total_automated_requirements": total_automated,
        "correct_predictions": correct_predictions,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "matches": results,
        "unmatched_baselines": unmatched_baselines,
        "unmatched_automated": list(unmatched_automated)
    }

    # Return project metrics to aggregate overall stats
    return precision, recall, f1_score, total_baselines, total_automated

def main():
    baseline_file = '/storage/scratch/6538142/setup2truth.json'  # Path to baseline JSON file
    results_dict = {}
    all_precisions, all_recalls, all_f1_scores = [], [], []
    total_baselines_sum, total_automated_sum = 0, 0

    # Mapping of automated JSON file paths to their corresponding project names
    project_files = {
        "/storage/scratch/6538142/jira21.json": "Jira_Performance_Testing_Tools",
        "/storage/scratch/6538142/lyrasis21.json": "Lyrasis Dura Cloud",
        "/storage/scratch/6538142/network_observability21.json": "Network_Observability",
        "/storage/scratch/6538142/openshift21.json": "OpenShift_UX_Product_Design",
        "/storage/scratch/6538142/qtdesign21.json": "Qt_Design_Studio",
        "/storage/scratch/6538142/redhat21.json": "Red_Hat_Developer_Website_v2"
    }

    # Loop through each project and compute comparisons
    for automated_file, project_name in project_files.items():
        print(f"Running comparison for: {project_name}")
        precision, recall, f1_score, total_baselines, total_automated = compare_files(
            baseline_file, automated_file, project_name, results_dict
        )

        # Collect statistics for overall summary
        all_precisions.append(precision)
        all_recalls.append(recall)
        all_f1_scores.append(f1_score)
        total_baselines_sum += total_baselines
        total_automated_sum += total_automated

    # Compute overall metrics
    overall_precision = sum(all_precisions) / len(all_precisions) if all_precisions else 0
    overall_recall = sum(all_recalls) / len(all_recalls) if all_recalls else 0
    overall_f1_score = sum(all_f1_scores) / len(all_f1_scores) if all_f1_scores else 0

    # Add summary at the top of the JSON output
    results_summary = {
        "summary": {
            "total_baselines": total_baselines_sum,
            "total_automated_requirements": total_automated_sum,
            "overall_precision": overall_precision,
            "overall_recall": overall_recall,
            "overall_f1_score": overall_f1_score
        }
    }
    
    # Merge summary with results
    results_summary.update(results_dict)

    # Save results
    output_file = "/storage/scratch/6538142/comparison21.json"
    with open(output_file, "w") as f:
        json.dump(results_summary, f, indent=4)
    
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
