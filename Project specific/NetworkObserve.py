import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

# Enable debugging mode
DEBUG = True  # Set to False to disable debugging prints

# Environment setup to manage GPU and temporary directories
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use only the first GPU
os.environ["HF_HOME"] = "/scratch/6538142/huggingface"  # Hugging Face cache
os.environ["TMPDIR"] = "/scratch/6538142/tmp"  # Temporary files in /scratch
torch.cuda.empty_cache()

# Model and tokenizer initialization
model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, token=os.getenv("HF_AUTH_TOKEN"))
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    token=os.getenv("HF_AUTH_TOKEN"),
    torch_dtype=torch.float16  # Use mixed precision to reduce memory usage
).to("cuda")  # Move the model directly to the GPU
tokenizer.pad_token = tokenizer.eos_token

# Configuration settings for generation
MAX_RESPONSE_LENGTH = 8192
REPEAT_PENALTY = 1.2

# Few-shot examples (kept intact as provided)
few_shot_examples = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Your role is to identify requirements from backlog items as a requirements engineer.
You are not allowed to rephrase or alter extracted text segments.
Identify and only return the unaltered segment from the text, without rephrasing or rewording, summarizing, or inferring any additional information.

Only extract text segments from a backlog item that fit the description of a requirement and its subcategories.
Definition of a requirement: A requirement specifies an added or modified functionality to the system or addresses a non-functional aspect such as quality attributes.
Subcategories:
- User-oriented functional requirement: Functionality experienced directly by the user.
- System-oriented functional requirement: Functionality implemented in the system that are not directly experienced by the user.
- Non-functional requirement: Specifies quality attributes such as usability, performance, or security.

Exclude:
- Documentation 
- Updates
- Feature enablement
- Rights or access 
- Tasks that have no direct impact on the system
- Suggestions
- Considerations
- Questions

List requirements as:
requirement 1: 
requirement 2: 
requirement 3:

### Example 1:
Backlog Item:
<|start_header_id|>user<|end_header_id|>
Summary: Plugin: localStorage
Description: Save selected column ids in local storage. Restore selection at first mount. By default, keep all columns selected if no previous configuration was found.
<|end_header_id|>

<|start_header_id|>assistant<|end_header_id|>
Requirement 1: Plugin: localStorage
Requirement 2: Save selected column ids in local storage
Requirement 3: Restore selection at first mount
<|end_header_id|>

### Example 2:
Backlog Item:
<|start_header_id|>user<|end_header_id|>
Summary: Use recommended kube labels
Description: There is a set of labels that are often used in kube applications by convention: https://kubernetes.io/docs/concepts/overview/working-with-objects/common-labels/#labels. Currently, the only label set on our pods/resources is app, which isn't in the "official" conventions but is often used for convenience (e.g., faster to type kubectl get pods -l app=goflow). We could use this set of labels: app.kubernetes.io/name, app.kubernetes.io/version, app.kubernetes.io/component, app.kubernetes.io/part-of, app.kubernetes.io/managed-by. We may also continue to use app for convenience.
<|end_header_id|>

<|start_header_id|>assistant<|end_header_id|>
Requirement 1: Use recommended kube labels
<|end_header_id|>

### Example 3:
Backlog Item:
<|start_header_id|>user<|end_header_id|>
Summary: Expose CNI type features as a config-map
Description: The console requires knowing the network type capabilities to show/hide some Network Policy form fields. The current logic is implemented as a features document inside the console code. This task aims to modify the current Cluster Network Operator to expose the network capabilities as an sdn-public Config Map, writable only by the SDN, readable by any system:authenticated user. Enhancement Proposal PR: https://github.com/openshift/enhancements/pull/875.
<|end_header_id|>

<|start_header_id|>assistant<|end_header_id|>
Requirement 1: Expose CNI type features as a config-map
Requirement 2: modify the current Cluster Network Operator to expose the network capabilities as an sdn-public Config Map, writable only by the SDN, readable by any system:authenticated user.
<|end_header_id|>

### Example 4:
Backlog Item:
<|start_header_id|>user<|end_header_id|>
Summary: Kube enrichment plugin
Description: We need to create (or reuse) a plugin for Kubernetes enrichment for fluentd (or any collector being part of the architecture). The existing plugin identifies pods through filenames when logs are processed, whereas we need to lookup pods from source and destination IPs within flow data. Our plugin can be tailored to our needs without unnecessary options. We need the following fields: src_pod, dst_pod, src_ns, dst_ns, src_wkd, dst_wkd (pod names, namespaces, and controllers). Acceptance Criteria: PR opened for Logging team to review, decision reached on solution, agreement on implementation, and subsequent stories created.
<|end_header_id|>

<|start_header_id|>assistant<|end_header_id|>
Requirement 1: Kube enrichment plugin
Requirement 2: We need to create (or reuse) a plugin for Kubernetes enrichment for fluentd (or any collector being part of the architecture).
<|end_header_id|>

### Example 5:
Backlog Item:
<|start_header_id|>user<|end_header_id|>
Summary: UI: Smart column sorting
Description: The column sorting algorithm should be based on the type of data. # For numeric values, it should sort by the number value.  For example, 2 should come before 10.  Note if an ASCII sort is mistakenly used, 10 would come before 2. # If the column is an IP address, it should sort by the four octets.  For example, 100.1.1.2 should come before 100.1.1.10 because the fourth octet is smaller.  Similarly, 100.1.5.200 should come before 100.1.10.1 because the third octet value is smaller. # The rest of the columns can be sorted based on the character set (ASCII sort).
<|end_header_id|>

<|start_header_id|>assistant<|end_header_id|>
Requirement 1: For numeric values, it should sort by the number value.  For example, 2 should come before 10.  Note if an ASCII sort is mistakenly used, 10 would come before 2.
Requirement 2: If the column is an IP address, it should sort by the four octets.  For example, 100.1.1.2 should come before 100.1.1.10 because the fourth octet is smaller.  Similarly, 100.1.5.200 should come before 100.1.10.1 because the third octet value is smaller.
Requirement 3: The rest of the columns can be sorted based on the character set (ASCII sort).
Requirement 4: The column sorting algorithm should be based on the type of data.
Requirement 5: UI: Smart column sorting.
<|end_header_id|>
"""

# Few-shot example requirements to exclude
few_shot_requirements = {
    "Use recommended kube labels",
    "Expose CNI type features as a config-map",
    "Kube enrichment plugin",
    "We need to create (or reuse) a plugin for Kubernetes enrichment for fluentd (or any collector being part of the architecture).",
    "Plugin: localStorage",
    "modify the current Cluster Network Operator to expose the network capabilities as an sdn-public Config Map, writable only by the SDN, readable by any system:authenticated user.",
    "Save selected column ids in local storage",
    "Restore selection at first mount",
    "For numeric values, it should sort by the number value.  For example, 2 should come before 10.  Note if an ASCII sort is mistakenly used, 10 would come before 2.",
    "If the column is an IP address, it should sort by the four octets.  For example, 100.1.1.2 should come before 100.1.1.10 because the fourth octet is smaller.  Similarly, 100.1.5.200 should come before 100.1.10.1 because the third octet value is smaller.",
    "The rest of the columns can be sorted based on the character set (ASCII sort).",
    "The column sorting algorithm should be based on the type of data.",
    "UI: Smart column sorting."

}

# Base prompt template
prompt_template = few_shot_examples + """
Backlog Item:
<|start_header_id|>user<|end_header_id|>
Summary: {summary}
Description: {description}
<|end_header_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

# Function to clean and extract only the requirement text
def clean_output(response):
    """
    Extracts and returns the cleaned requirement text, removing prefixes and numbering.
    """
    lines = response.split("\n")
    requirements = [
        line.strip() for line in lines
        if line.strip().startswith("Requirement")
    ]
    cleaned_requirements = []
    for req in requirements:
        # Extract the text after the prefix 'requirement X: '
        requirement_text = req.split(": ", 1)[-1].strip()
        if requirement_text not in few_shot_requirements:
            cleaned_requirements.append(requirement_text)
    return cleaned_requirements


file_pairs = [
    #("/storage/scratch/6538142/Cost_Management.xlsx", "/storage/scratch/6538142/costmanagement1.json"),
    #("/storage/scratch/6538142/Jira_Performance_Testing_Tools.xlsx", "/storage/scratch/6538142/jira1.json"),
    #("/storage/scratch/6538142/Lyrasis Dura Cloud.xlsx", "/storage/scratch/6538142/lyrasis1.json"),
    ("/storage/scratch/6538142/Network_Observability.xlsx", "/storage/scratch/6538142/network_observability21.json"),
    #("/storage/scratch/6538142/OpenShift_UX_Product_Design.xlsx", "/storage/scratch/6538142/openshift1.json"),
    #("/storage/scratch/6538142/Qt_Design_Studio.xlsx", "/storage/scratch/6538142/qtdesign1.json"),
    #("/storage/scratch/6538142/Red_Hat_Developer_Website_v2.xlsx", "/storage/scratch/6538142/redhat1.json")
]

# Process each file pair
for input_file, output_file in file_pairs:
    df = pd.read_excel(input_file)  # Load the Excel file
    all_requirements_with_ids = []  # Store each requirement with its corresponding ID

    # Process each backlog item individually
    for idx, row in df.iterrows():
        summary = row.get('summary', '')
        description = row.get('description', '')
        requirement_id = row.get('id')  # Extract the corresponding ID

        # Format the input prompt
        prompt = prompt_template.format(summary=summary, description=description)
        
        # Tokenize the input prompt
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        # Generate the model response
        outputs = model.generate(
            inputs.input_ids,
            max_length=MAX_RESPONSE_LENGTH,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,  # Disable randomness
            temperature=0.0,  # Force deterministic output
            repetition_penalty=REPEAT_PENALTY  # Discourage repetitive text
        )
        
        # Decode and clean the response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        requirements = clean_output(response)
        
        # Add each requirement along with its ID
        for requirement in requirements:
            all_requirements_with_ids.append({
                "requirement": requirement,
                "id": int(requirement_id) if pd.notna(requirement_id) else None  # Handle missing IDs
            })

        # Save requirements after processing each backlog item
        with open(output_file, 'w') as f:
            json.dump(all_requirements_with_ids, f, indent=4)
    print(f"Processed {input_file} and saved requirements to {output_file}")
