import json
import pandas as pd
import re

def read_json_requirements(json_file):
    """Reads requirements from a JSON file."""
    try:
        with open(json_file, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except Exception as e:
        print(f"Error reading {json_file}: {e}")
        return {}

def read_excel_requirements(excel_file):
    """Reads description, summary, and id columns from an Excel file."""
    try:
        df = pd.read_excel(excel_file)
        # Ensure necessary columns exist
        if {'description', 'summary', 'id'}.issubset(df.columns):
            df['combined_text'] = df['description'].astype(str) + " " + df['summary'].astype(str)
            return df[['id', 'combined_text']]
        else:
            print(f"Missing required columns in {excel_file}")
            return pd.DataFrame()
    except Exception as e:
        print(f"Error reading {excel_file}: {e}")
        return pd.DataFrame()

def link_requirements_to_ids(json_file, excel_files):
    """Links JSON requirements to their IDs from Excel files."""
    json_data = read_json_requirements(json_file)
    results = {}

    for project_name, excel_file in excel_files.items():
        project_key = f"\\\\{project_name}"
        requirements = json_data.get(project_key, [])
        excel_data = read_excel_requirements(excel_file)

        if excel_data.empty:
            print(f"No data found for project: {project_name}")
            continue

        # Match requirements
        project_results = []
        for requirement in requirements:
            # Escape special characters in the requirement string
            try:
                match = excel_data[excel_data['combined_text'].str.contains(re.escape(requirement), case=False, na=False, regex=True)]
                if not match.empty:
                    matched_id = match.iloc[0]['id']
                    project_results.append({"requirement": requirement, "id": matched_id})
                else:
                    project_results.append({"requirement": requirement, "id": None})  # No match found
            except Exception as e:
                print(f"Error matching requirement '{requirement}': {e}")
                project_results.append({"requirement": requirement, "id": None})

        results[project_name] = project_results

    return results

def save_results_to_json(results, output_file):
    """Saves the results to a JSON file."""
    try:
        # Convert all values to native Python types (e.g., int64 -> int)
        cleaned_results = {
            project: [
                {
                    "requirement": item["requirement"],
                    "id": int(item["id"]) if item["id"] is not None else None
                }
                for item in project_data
            ]
            for project, project_data in results.items()
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(cleaned_results, f, indent=4, ensure_ascii=False)
        print(f"Results saved to {output_file}")
    except Exception as e:
        print(f"Error saving results to {output_file}: {e}")


def main():
    json_file = '/Users/marcrademakers/Desktop/RequirementID/requirement_linking_results.json'
    excel_files = {
        "Cost_Management": "/Users/marcrademakers/Desktop/select_sample/samples/Cost_Management.xlsx",
        "Jira_Performance_Testing_Tools": "/Users/marcrademakers/Desktop/select_sample/samples/Jira_Performance_Testing_Tools.xlsx",
        "Lyrasis Dura Cloud": "/Users/marcrademakers/Desktop/select_sample/samples/Lyrasis Dura Cloud.xlsx",
        "Network_Observability": "/Users/marcrademakers/Desktop/select_sample/samples/Network_Observability.xlsx",
        "OpenShift_UX_Product_Design": "/Users/marcrademakers/Desktop/select_sample/samples/OpenShift_UX_Product_Design.xlsx",
        "Qt_Design_Studio": "/Users/marcrademakers/Desktop/select_sample/samples/Qt_Design_Studio.xlsx",
        "Red_Hat_Developer_Website_v2": "/Users/marcrademakers/Desktop/select_sample/samples/Red_Hat_Developer_Website_v2.xlsx"
    }
    output_file = '/Users/marcrademakers/Desktop/RequirementID/requirement_linking_results.json'
    
    results = link_requirements_to_ids(json_file, excel_files)
    save_results_to_json(results, output_file)

if __name__ == "__main__":
    main()
