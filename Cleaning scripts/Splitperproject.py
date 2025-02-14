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

def process_txt_files(file_paths, delimiter="<Files"):
    """
    Process multiple text files and organize their content into a dictionary.

    Args:
        file_paths (list): List of file paths to process.
        delimiter (str): String to split content into sections. Defaults to "<Files".

    Returns:
        dict: A dictionary with project names as keys and raw tags as values.
    """
    project_dict = defaultdict(list)

    for file_path in file_paths:
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

def save_as_json(project_dict, output_file):
    """
    Save the project dictionary as a JSON file.
    """
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(project_dict, file, ensure_ascii=False, indent=4)
    print(f"Dictionary saved as JSON to {output_file}")

def load_from_json(input_file):
    """
    Load a dictionary from a JSON file.
    """
    with open(input_file, 'r', encoding='utf-8') as file:
        return json.load(file)

# Main Script
if __name__ == "__main__":
    # Define file paths
    file_paths = [
    '/Users/marcrademakers/Desktop/References/high_nfr.txt',
    '/Users/marcrademakers/Desktop/References/high_system.txt',
    '/Users/marcrademakers/Desktop/References/high_user.txt',
    '/Users/marcrademakers/Desktop/References/low_nfr.txt',
    '/Users/marcrademakers/Desktop/References/low_system.txt',
    '/Users/marcrademakers/Desktop/References/low_user.txt',
    '/Users/marcrademakers/Desktop/References/medium_nfr.txt',
    '/Users/marcrademakers/Desktop/References/medium_system.txt',
    '/Users/marcrademakers/Desktop/References/medium_user.txt',
    ]
    
    # Process files
    project_data = process_txt_files(file_paths)
    
    # Save the dictionary as JSON
    json_file_path = "./references.json"
    save_as_json(project_data, json_file_path)
    
    # Reload the dictionary from JSON (to demonstrate loading)
    reloaded_data = load_from_json(json_file_path)
    print("Reloaded Dictionary:")
    print(json.dumps(reloaded_data, indent=4, ensure_ascii=False))
