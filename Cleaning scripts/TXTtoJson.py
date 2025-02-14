import os
import json
from collections import defaultdict
import chardet  # To detect encoding

def detect_encoding(file_path):
    """
    Detect the encoding of a file.
    """
    with open(file_path, 'rb') as file:
        raw_data = file.read(10000)  # Read a portion of the file
    return chardet.detect(raw_data)['encoding']

def process_txt_file(file_path, delimiter="<Files"):
    """
    Process a single text file and organize its content into a dictionary.

    Args:
        file_path (str): Path of the file to process.
        delimiter (str): String to split content into sections. Defaults to "<Files".

    Returns:
        dict: A dictionary with project names as keys and raw tags as values.
    """
    project_dict = defaultdict(list)
    
    try:
        # Detect encoding of the file
        encoding = detect_encoding(file_path)
        
        # Read file content with detected encoding
        with open(file_path, 'r', encoding=encoding) as file:
            content = file.read()
        
        # Split content into sections using the delimiter
        sections = content.split(delimiter)[1:]  # Skip the first empty split
        
        for section in sections:
            # Extract project name and its content
            project_name, _, section_content = section.partition("> - §")
            project_name = project_name.strip()
            
            # Extract the raw tags under each reference
            references = section_content.split("Reference ")
            for ref in references[1:]:  # Skip the first split (non-reference content)
                raw_tag = ref.split("\n", 1)[1].strip()  # Extract the line after "Reference"
                project_dict[project_name].append(raw_tag)
    
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
    
    return project_dict

def save_as_json(data, output_file):
    """
    Save the processed data as a JSON file.

    Args:
        data (dict): The data to save.
        output_file (str): Path to save the JSON file.
    """
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
    print(f"Saved JSON to {output_file}")

# Main Script
if __name__ == "__main__":
    # Define file groups with granularity and category
    file_paths = {
        "high_nfr": "/Users/marcrademakers/Desktop/Tagged data/high_nfr.txt",
        "medium_nfr": "/Users/marcrademakers/Desktop/Tagged data/medium_nfr.txt",
        "low_nfr": "/Users/marcrademakers/Desktop/Tagged data/low_nfr.txt",
        "high_system": "/Users/marcrademakers/Desktop/Tagged data/high_system.txt",
        "medium_system": "/Users/marcrademakers/Desktop/Tagged data/medium_system.txt",
        "low_system": "/Users/marcrademakers/Desktop/Tagged data/low_system.txt",
        "high_user": "/Users/marcrademakers/Desktop/Tagged data/high_user.txt",
        "medium_user": "/Users/marcrademakers/Desktop/Tagged data/medium_user.txt",
        "low_user": "/Users/marcrademakers/Desktop/Tagged data/low_user.txt",
        "high_nfr_2": "/Users/marcrademakers/Desktop/Tagged data 2/high_nfr.txt",
        "medium_nfr_2": "/Users/marcrademakers/Desktop/Tagged data 2/medium_nfr.txt",
        "low_nfr_2": "/Users/marcrademakers/Desktop/Tagged data 2/low_nfr.txt",
        "high_system_2": "/Users/marcrademakers/Desktop/Tagged data 2/high_system.txt",
        "medium_system_2": "/Users/marcrademakers/Desktop/Tagged data 2/medium_system.txt",
        "low_system_2": "/Users/marcrademakers/Desktop/Tagged data 2/low_system.txt",
        "high_user_2": "/Users/marcrademakers/Desktop/Tagged data 2/high_user.txt",
        "medium_user_2": "/Users/marcrademakers/Desktop/Tagged data 2/medium_user.txt",
        "low_user_2": "/Users/marcrademakers/Desktop/Tagged data 2/low_user.txt",
    }

    # Process and save files for each category and granularity
    for key, path in file_paths.items():
        try:
            # Process file
            data = process_txt_file(path)
            
            # Save JSON in the same directory as the source file
            output_dir = os.path.dirname(path)
            output_file = os.path.join(output_dir, f"{key}.json")
            save_as_json(data, output_file)
        
        except Exception as e:
            print(f"Failed to process {key}: {e}")
