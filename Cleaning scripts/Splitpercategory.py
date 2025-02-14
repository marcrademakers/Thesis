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

def process_txt_files(file_paths, category_name, data_dict):
    """
    Process multiple text files and organize their content into a dictionary.

    Args:
        file_paths (list): List of file paths to process.
        category_name (str): Name of the category to store data under.
        data_dict (dict): Dictionary to store parsed data.
    """
    delimiter = "<Files"  # Define delimiter for splitting sections

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
                    data_dict[category_name][project_name].append(raw_tag)
        
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

def save_as_json(data_dict, output_file):
    """
    Save the structured data as a JSON file.
    """
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(data_dict, file, ensure_ascii=False, indent=4)
    print(f"Data saved as JSON to {output_file}")

# Main Script
if __name__ == "__main__":
    # Define file paths and corresponding category names '/Users/marcrademakers/Desktop/Tagged data' '/Users/marcrademakers/Desktop/Tagged data 2'
    file_categories = {
        "high_nfr": ['/Users/marcrademakers/Desktop/Tagged data 2/high_nfr.txt'],
        "low_nfr": ['/Users/marcrademakers/Desktop/Tagged data 2/low_nfr.txt'],
        "medium_nfr": ['/Users/marcrademakers/Desktop/Tagged data 2/medium_nfr.txt'],
        "high_system": ['/Users/marcrademakers/Desktop/Tagged data 2/high_system.txt'],
        "low_system": ['/Users/marcrademakers/Desktop/Tagged data 2/low_system.txt'],
        "medium_system": ['/Users/marcrademakers/Desktop/Tagged data 2/medium_system.txt'],
        "high_user": ['/Users/marcrademakers/Desktop/Tagged data 2/high_user.txt'],
        "low_user": ['/Users/marcrademakers/Desktop/Tagged data 2/low_user.txt'],
        "medium_user": ['/Users/marcrademakers/Desktop/Tagged data 2/medium_user.txt'],
    }

    # Initialize dictionary structure
    consolidated_data = {category: defaultdict(list) for category in file_categories}

    # Process each category
    for category, files in file_categories.items():
        process_txt_files(files, category, consolidated_data)

    # Save the consolidated JSON file
    save_as_json(consolidated_data, '/Users/marcrademakers/Desktop/RQ4/consolidated_data.json')
