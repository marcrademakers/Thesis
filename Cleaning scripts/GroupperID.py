import json

def read_json_requirements(json_file):
    """Reads requirements from a JSON file."""
    try:
        with open(json_file, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except Exception as e:
        print(f"Error reading {json_file}: {e}")
        return {}

def group_requirements_by_id(baseline_data):
    """
    Groups requirements by their IDs while maintaining project structure.
    
    Args:
        baseline_data (dict): The baseline JSON data.
        
    Returns:
        dict: A nested dictionary with projects as keys, and IDs as sub-keys containing grouped requirements.
    """
    grouped_data = {}

    for project, requirements in baseline_data.items():
        grouped_data[project] = {}
        for requirement in requirements:
            req_id = requirement["id"]
            req_text = requirement["requirement"]
            
            if req_id not in grouped_data[project]:
                grouped_data[project][req_id] = []
            
            grouped_data[project][req_id].append(req_text)
    
    return grouped_data

def save_grouped_data_to_json(grouped_data, output_file):
    """Saves the grouped data into a JSON file."""
    try:
        with open(output_file, 'w', encoding='utf-8') as file:
            json.dump(grouped_data, file, indent=4)
        print(f"Grouped data saved to {output_file}")
    except Exception as e:
        print(f"Error saving grouped data: {e}")

def main():
    baseline_file = '/Users/marcrademakers/Desktop/RequirementID/requirement_linking_results.json'  # Input file
    output_file = '/Users/marcrademakers/Desktop/RequirementID/grouped_requirements.json'  # Output file

    # Read and process the baseline file
    baseline_data = read_json_requirements(baseline_file)
    grouped_data = group_requirements_by_id(baseline_data)

    # Save the grouped data
    save_grouped_data_to_json(grouped_data, output_file)

if __name__ == "__main__":
    main()
