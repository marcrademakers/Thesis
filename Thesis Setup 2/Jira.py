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
Summary: Install Jira distribution and download jira home in parallel
Description: Aws infrastructure could provide running instances faster if we download dataset and jira installer in parallel.
<|end_header_id|>

<|start_header_id|>assistant<|end_header_id|>
Requirement 1: Install Jira distribution and download jira home in parallel
Requirement 2: Aws infrastructure could provide running instances faster if we download dataset and jira installer in parallel.
<|end_header_id|>

### Example 2:
Backlog Item:
<|start_header_id|>user<|end_header_id|>
Summary: Make legend for JPT charts more usable
Description: Currently it is very hard to find the data for experiment and baseline for the same scenario/dataset. I suggest that we move the chart legend to the side (to align the items horizontally) and sort them by the dataset label, hence putting experiment under the baseline (or the other way round, the main point is for them to stick together).
<|end_header_id|>

<|start_header_id|>assistant<|end_header_id|>
Requirement 1: Make legend for JPT charts more usable
<|end_header_id|>

### Example 3:
Backlog Item:
<|start_header_id|>user<|end_header_id|>
Summary: Customise stack creation timeouts
Description: Sometimes stack creation times out with a 30-minute timeout, especially when creating many stacks at once. It is possible to define stack creation timeout from Stack API from aws-resources, but not possible to specify in either StandaloneFormula nor DataCenterFormula in aws-infrastructure.
<|end_header_id|>

<|start_header_id|>assistant<|end_header_id|>

<|end_header_id|>

### Example 4:
Backlog Item:
<|start_header_id|>user<|end_header_id|>
Summary: Provide builders in infrastructure
Description: In JPT 1.0 we started with Kotlin defaults and named parameters. Unfortunately, it caused a lot of API breakages, and this doesn't work with Java well. We have abandoned defaults, and now we have very complex objects to create like JiraNodeConfig: {{noformat}} JiraNodeConfig( debug = DisabledJvmDebug(), name = "jira-node", jvmArgs = JiraJvmArgs(), launchTimeouts = JiraLaunchTimeouts( offlineTimeout = Duration.ofMinutes(8), initTimeout = Duration.ofMinutes(4), upgradeTimeout = Duration.ofMinutes(8), unresponsivenessTimeout = Duration.ofMinutes(4) ), splunkForwarder = DisabledSplunkForwarder(), remoteJmx = EnabledRemoteJmx() ) {{noformat}} We also have to deprecate constructors each time we add a new field to the object. It leads to extra effort for both JPT development and library upgrades. We can provide Builders with reasonable default parameters: - No upgrade/deprecation pain when extending the class. - Readable object construction in Java (it's hard to read constructors with multiple parameters; a builder will add kind of named parameters feature to the Java world). - Easier to use. - We can share defaults that make sense.
<|end_header_id|>

<|start_header_id|>assistant<|end_header_id|>
Requirement 1: Provide builders in infrastructure
<|end_header_id|>

### Example 5:
Backlog Item:
<|start_header_id|>user<|end_header_id|>
Summary: Extract infrastructure integration tests to a sub-module
Description: I want to use `docker-infrastructure` in infrastructure's tests. (infrastructure depends on docker-infrastructure). I want to use `infrastructure` in `docker-infrastructure` the same way we use it in `aws-infrastructure` (Installing Java, Product etc.). I want to avoid circular dependency by extracting integration tests to a separate module. IMO it will also simplify infrastructure's `build.gradle.kts`. It wouldn't have to separate tests by class name and to introduce `testIntegration` task. Project build would be responsible for running IT.
<|end_header_id|>

<|start_header_id|>assistant<|end_header_id|>
Requirement 1: I want to use `docker-infrastructure` in infrastructure's tests. (infrastructure depends on docker-infrastructure).
Requirement 2: I want to use `infrastructure` in `docker-infrastructure` the same way we use it in `aws-infrastructure` (Installing Java, Product etc.).
Requirement 3: Extract infrastructure integration tests to a sub-module.
<|end_header_id|>
"""

# Few-shot example requirements to exclude
few_shot_requirements = {
    "Make legend for JPT charts more usable",
    "Customise stack creation timeouts",
    "Provide builders in infrastructure",
    "Install Jira distribution and download jira home in parallel",
    "Aws infrastructure could provide running instances faster if we download dataset and jira installer in parallel.",
    "I want to use `docker-infrastructure` in infrastructure's tests. (infrastructure depends on docker-infrastructure).",
    "I want to use `infrastructure` in `docker-infrastructure` the same way we use it in `aws-infrastructure` (Installing Java, Product etc.).",
    "Extract infrastructure integration tests to a sub-module."

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
    ("/storage/scratch/6538142/Jira_Performance_Testing_Tools.xlsx", "/storage/scratch/6538142/jira21.json"),
    #("/storage/scratch/6538142/Lyrasis Dura Cloud.xlsx", "/storage/scratch/6538142/lyrasis1.json"),
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
        print("Raw model response:\n", response)

        requirements = clean_output(response)
        print("Extracted Requirements:", requirements)

        
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

    
