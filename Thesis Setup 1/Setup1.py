import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

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

# Few-shot examples
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
Summary: Consolidate query_table properties
Description: h1. User Story
As a developer I want less code duplication so that it is easier to track.
----
h2. Backend Requirements
* Consolidate the various query_table properties in the query handlers. 

<|end_header_id|>
<|start_header_id|>assistant<|end_header_id|>

requirement 1: Consolidate query_table properties
requirement 2: As a developer I want less code duplication so that it is easier to track.
requirement 3: Consolidate the various query_table properties in the query handlers.

<|end_header_id|>

### Example 2:
Backlog Item:
<|start_header_id|>user<|end_header_id|>
Summary: Record manifest generation date for all providers
Description: As a user I want to know the date that my data source report was generated. Knowing this date will give me the missing context for what data my Cost Management dashboard is showing and will give me a good sync point to verify Cost Management with other external tools. 
<|end_header_id|>

<|start_header_id|>assistant<|end_header_id|>
requirement 1: Record manifest generation date for all providers
requirement 2: As a user I want to know the date that my data source report was generated.
requirement 3: Report generated date is saved for each manifest
<|end_header_id|>

### Example 3:
Backlog Item:
<|start_header_id|>user<|end_header_id|>
Summary: Adding support for hierarchies in clouds
Description: h1. Feature overview Provide understanding and grouping for hierarchical accounts coming from clouds. h1. Goals * Provide support for hierarchies in the clouds supported. * Allow the customer to see charages for one element on the hierarchy (i.e. an AWS Organizational Unit and everything associated to it up to the end of the hierarchy) * Allow the administrator to define those hierarchies for RBAC, allowing a customer to see one level of the hierarchy and everything below (in contrast to defining a list of accounts) h1. Background Accounts in the cloud can often be hierarchical. Depending on the business of the customer, many accounts can be organized to reflect better the requirements from the business. AWS currently provides different organizations. In the basic one, a customer can have a master account (that accrues all data from the different accounts), and member (linked) account. However, that is not enough for customer, so clouds offer additional ways of organizing their accounts into hierarchies. Amazon also provides a way of organizing accounts together for management purposes (not billing). You can use organizational units (OUs) to group accounts together to administer as a single unit. This greatly simplifies the management of your accounts. For example, you can attach a policy-based control to an OU, and all accounts within the OU automatically inherit the policy. Reflecting this organization into the account and RBAC of cost management can greatly reduce the requirements for organization inside cost management and thus make things easier for customers.
<|end_header_id|>

<|start_header_id|>assistant<|end_header_id|>
requirement 1: Provide understanding and grouping for hierarchical accounts coming from clouds.
requirement 2: Adding support for hierarchies in clouds
requirement 3: Provide support for hierarchies in the clouds supported.
requirement 4: Allow the customer to see charages for one element on the hierarchy (i.e. an AWS Organizational Unit and everything associated to it up to the end of the hierarchy)
requirement 5: Allow the administrator to define those hierarchies for RBAC, allowing a customer to see one level of the hierarchy and everything below (in contrast to defining a list of accounts)
<|end_header_id|>

### Example 4:
Backlog Item:
<|start_header_id|>user<|end_header_id|>
Summary: S3 Big Data Pipeline
Description: h1. User Story As developers we want cost management data stored in S3 and processed using a big data tool so that we can store data for longer periods and process large amounts of data more efficiently. h2. Prioritization / Business Case * We have already done spike and PoC work toward this objective, this is just aiming for a complete solution where we process and store data using S3 and a big data tool * Scale and process more customers h2. Out Of Scope * Although this will enable longer term storage, this is just getting the infrastructure in place, not actually doing work for anything beyond 2 months of data in the API/UI h2. Impacts * API * Data Engineering * Database h2. Related Stories * Deploy Presto in OpenShift * Run Presto within Docker-Compose * Configure Presto + Hive with our S3 * Dynamically create Presto tables for S3 data * Trigger summarization on S3 events * Unit test generation flow * Convert AWS Summary SQL to Presto * Convert Azure Summary SQL to Presto * Convert OpenShift Summary SQL to Presto * Convert OpenShift on AWS Summary SQL to Presto * Convert OpenShift on Azure Summary SQL to Presto * Calculate cost model cost using Presto * Load summarized data into Postgres * Presto monitoring h2. External Dependencies * OpenShift Quota * Presto image from metering team (regular sync of version; like Python or Django) h1. Documentation Requirements * We might want to document how we store users data and highlight steps taken to ensure security h1. Backend Requirements * Are there any prerequisites required before working this epic? No h1. QE Requirements * Does QE need specific data or tooling to successfully test this epic? S3 Bucket for ephemeral environments (just use path structure)? h1. Release Criteria * Can this epic be released as individual issues/tasks are completed? Partially, infrastructure can be deployed but summarization should switch flows from Postgresql to Presto in a single deployment * Can the backend be released without the frontend? Yes * Has QE approved this epic? Yes.
<|end_header_id|>

<|start_header_id|>assistant<|end_header_id|>
requirement 1: S3 Big Data Pipeline
requirement 2: As developers we want cost management data stored in S3 and processed using a big data tool so that we can store data for longer periods and process large amounts of data more efficiently.
<|end_header_id|>

### Example 5:
Backlog Item:
<|start_header_id|>user<|end_header_id|>
Summary: Send Data to S3
Description: h1. User Story As a data engineer I want to ship our data to object storage so that it is available to be sent off to other sources (e.g. the DataHub), we utilize cheap long term storage, and so we can trigger events off of data entering object storage. ---- h2. UX Requirements * N/A h2. UI Requirements * N/A h2. Documentation Requirements * N/A h2. Backend Requirements * We will want to send report files to S3/Ceph as we get new manifests. h2. QE Requirements * h2. Additional Information and Assumptions * h2. Acceptance Criteria * As data is ingressed to cost it should be placed on S3 in a \"data/csv/(account)/(provider_uuid)/(year)/(month)/\" format * All OCP files should be placed in the report month directory (will contain some overlapping data) * All Azure files should be placed in the report month directory (they are daily) * The latest AWS files should be placed in the report month directory (report download are full month), previous report files downloaded for the month should be removed
<|end_header_id|>

<|start_header_id|>assistant<|end_header_id|>
Requirement 1: We will want to send report files to S3/Ceph as we get new manifests.
Requirement 2: * As data is ingressed to cost it should be placed on S3 in a \"data/csv/(account)/(provider_uuid)/(year)/(month)/\" format
Requirement 3: All OCP files should be placed in the report month directory (will contain some overlapping data).
Requirement 4: All Azure files should be placed in the report month directory (they are daily).
Requirement 5: The latest AWS files should be placed in the report month directory (report download are full month), previous report files downloaded for the month should be removed.
Requirement 6: As a data engineer I want to ship our data to object storage so that it is available to be sent off to other sources (e.g. the DataHub), we utilize cheap long term storage, and so we can trigger events off of data entering object storage.
Requirement 7: Send Data to S3.
<|end_header_id|>


"""

# Few-shot example requirements to exclude
few_shot_requirements = {
    "Consolidate query_table properties",
    "As a developer I want less code duplication so that it is easier to track.",
    "Consolidate the various query_table properties in the query handlers.",
    "Record manifest generation date for all providers",
    "As a user I want to know the date that my data source report was generated.",
    "Report generated date is saved for each manifest",
    "Provide understanding and grouping for hierarchical accounts coming from clouds.",
    "Adding support for hierarchies in clouds",
    "Provide support for hierarchies in the clouds supported.",
    "Allow the customer to see charages for one element on the hierarchy (i.e. an AWS Organizational Unit and everything associated to it up to the end of the hierarchy)",
    "Allow the administrator to define those hierarchies for RBAC, allowing a customer to see one level of the hierarchy and everything below (in contrast to defining a list of accounts)",
    "S3 Big Data Pipeline",
    "As developers we want cost management data stored in S3 and processed using a big data tool so that we can store data for longer periods and process large amounts of data more efficiently.",
    "We will want to send report files to S3/Ceph as we get new manifests.",
    "* As data is ingressed to cost it should be placed on S3 in a \"data/csv/(account)/(provider_uuid)/(year)/(month)/\" format",
    "All OCP files should be placed in the report month directory (will contain some overlapping data)",
    "All Azure files should be placed in the report month directory (they are daily)",
    "The latest AWS files should be placed in the report month directory (report download are full month), previous report files downloaded for the month should be removed",
    "As a data engineer I want to ship our data to object storage so that it is available to be sent off to other sources (e.g. the DataHub), we utilize cheap long term storage, and so we can trigger events off of data entering object storage.",
    "Send Data to S3"

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
        if line.strip().startswith("requirement")
    ]
    cleaned_requirements = []
    for req in requirements:
        # Extract the text after the prefix 'requirement X: '
        requirement_text = req.split(": ", 1)[-1].strip()
        if requirement_text not in few_shot_requirements:
            cleaned_requirements.append(requirement_text)
    return cleaned_requirements

file_pairs = [
    #("/storage/scratch/6538142/Cost_Management.xlsx", "/storage/scratch/6538142/setup2costmanagementtest.json"),
    ("/storage/scratch/6538142/Jira_Performance_Testing_Tools.xlsx", "/storage/scratch/6538142/jira11.json"),
    ("/storage/scratch/6538142/Lyrasis Dura Cloud.xlsx", "/storage/scratch/6538142/lyrasis11.json"),
    ("/storage/scratch/6538142/Network_Observability.xlsx", "/storage/scratch/6538142/network_observability11.json"),
    ("/storage/scratch/6538142/OpenShift_UX_Product_Design.xlsx", "/storage/scratch/6538142/openshift11.json"),
    ("/storage/scratch/6538142/Qt_Design_Studio.xlsx", "/storage/scratch/6538142/qtdesign11.json"),
    ("/storage/scratch/6538142/Red_Hat_Developer_Website_v2.xlsx", "/storage/scratch/6538142/redhat11.json")
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
