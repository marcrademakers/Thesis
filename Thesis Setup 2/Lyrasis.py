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
Summary: UI: Remove upload tool applet, reinstate single file upload
Description: Now that there is a solid UI as part of the Sync Tool, and because of the seemingly constant issues with java applet technology, the multi-file upload applet should be removed from DurAdmin and the single file upload option should be re-instated. This work should occur alongside DURACLOUD-775, which provides easy access to the Sync Tool.
<|end_header_id|>

<|start_header_id|>assistant<|end_header_id|>

<|end_header_id|>

### Example 2:
Backlog Item:
<|start_header_id|>user<|end_header_id|>
Summary: UI: Handle Glacier storage state errors gracefully
Description: When content is stored in Glacier, attempts to download, view, copy, or edit the properties of that content will result in an error (http status 409). This task is to provide users with useful feedback when they attempt any of these actions on a content item in Glacier. More specifically, when a call is made on a Glacier content item via the storeclient, a ContentStateException is thrown. (If the content item was recently added or has been retrieved from Glacier, it will simply download as expected) This task is to show the user an appropriate error message indicating that the content is in Glacier. For now, that error message can indicate that the user should contact DuraCloud support if they wish to retrieve that content item.
<|end_header_id|>

<|start_header_id|>assistant<|end_header_id|>
Requirement 1: UI: Handle Glacier storage state errors gracefully
Requirement 2: This task is to provide users with useful feedback when they attempt any of these actions on a content item in Glacier. More specifically, when a call is made on a Glacier content item via the storeclient, a ContentStateException is thrown. (If the content item was recently added or has been retrieved from Glacier, it will simply download as expected) This task is to show the user an appropriate error message indicating that the content is in Glacier. For now, that error message can indicate that the user should contact DuraCloud support if they wish to retrieve that content item.
<|end_header_id|>

### Example 3:
Backlog Item:
<|start_header_id|>user<|end_header_id|>
Summary: SyncTool --dryrun
Description: This improvement is to create a dryrun flag on the synctool. This flag would tell the synctool to go through the steps of determining which files should be sync-ed up to DuraCloud (including deleted and modified content), but instead of actually performing the sync, just a report will be produced detailing which content would have been sync-ed.
<|end_header_id|>

<|start_header_id|>assistant<|end_header_id|>
Requirement 1: This improvement is to create a dryrun flag on the synctool. This flag would tell the synctool to go through the steps of determining which files should be sync-ed up to DuraCloud (including deleted and modified content), but instead of actually performing the sync, just a report will be produced detailing which content would have been sync-ed.
<|end_header_id|>

### Example 4:
Backlog Item:
<|start_header_id|>user<|end_header_id|>
Summary: SystemUserCredential Across Webapps
Description: This improvement is to create the SystemUserCredential in a similar was to how RootUserCredential is created, via a default username/password over-written by system properties. Once this is done, DuradminExecutorImpl.java should be evaluated for exchanging root-user with system-user.
<|end_header_id|>

<|start_header_id|>assistant<|end_header_id|>
Requirement 1: SystemUserCredential Across Webapps
Requirement 2: This improvement is to create the SystemUserCredential in a similar was to how RootUserCredential is created, via a default username/password over-written by system properties. Once this is done, DuradminExecutorImpl.java should be evaluated for exchanging root-user with system-user.
<|end_header_id|>

### Example 5:
Backlog Item:
<|start_header_id|>user<|end_header_id|>
Summary: StoreClient encryption support
Description: This feature is to provide a means for users to encrypt/decrypt content stored and retrieved through the StoreClient.  The StoreClient should have a constructor that optionally takes a symmetric or asymmetric encryption key.
<|end_header_id|>

<|start_header_id|>assistant<|end_header_id|>
Requirement 1: This feature is to provide a means for users to encrypt/decrypt content stored and retrieved through the StoreClient.  The StoreClient should have a constructor that optionally takes a symmetric or asymmetric encryption key.
Requirement 2: StoreClient encryption support.
<|end_header_id|>

"""

# Few-shot example requirements to exclude
few_shot_requirements = {
    "UI: Handle Glacier storage state errors gracefully",
    "This task is to provide users with useful feedback when they attempt any of these actions on a content item in Glacier. More specifically, when a call is made on a Glacier content item via the storeclient, a ContentStateException is thrown. (If the content item was recently added or has been retrieved from Glacier, it will simply download as expected) This task is to show the user an appropriate error message indicating that the content is in Glacier. For now, that error message can indicate that the user should contact DuraCloud support if they wish to retrieve that content item.",
    "This improvement is to create a dryrun flag on the synctool. This flag would tell the synctool to go through the steps of determining which files should be sync-ed up to DuraCloud (including deleted and modified content), but instead of actually performing the sync, just a report will be produced detailing which content would have been sync-ed.",
    "SystemUserCredential Across Webapps",
    "This improvement is to create the SystemUserCredential in a similar was to how RootUserCredential is created, via a default username/password over-written by system properties. Once this is done, DuradminExecutorImpl.java should be evaluated for exchanging root-user with system-user.",
    "This feature is to provide a means for users to encrypt/decrypt content stored and retrieved through the StoreClient.  The StoreClient should have a constructor that optionally takes a symmetric or asymmetric encryption key.",
    "StoreClient encryption support."

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
    ("/storage/scratch/6538142/Lyrasis Dura Cloud.xlsx", "/storage/scratch/6538142/lyrasis21.json"),
    #("/storage/scratch/6538142/Network_Observability.xlsx", "/storage/scratch/6538142/network_observability1.json"),
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
