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
Summary: Gizmo design for rotate
Description: Create design for a gizmo that is used to rotate 3D objects and define how interaction works: - How gizmo is activated and how active gizmo is indicated. - Icon needed too. - UI design for gizmo + creation of gizmo. - Item selection and manipulation interaction.
<|end_header_id|>

<|start_header_id|>assistant<|end_header_id|>
Requirement 1: Gizmo design for rotate
<|end_header_id|>

### Example 2:
Backlog Item:
<|start_header_id|>user<|end_header_id|>
Summary: Add support for uip / uia import
Description: What is needed: User must be able to import UIA and UIP files. Who needs this: DS User. Why it's needed: To have a proper migration path from 3DS to DS. Acceptance criteria: - User can import uip and uia files using import dialog. - Options specific for uip and uia are visible in the dialog using the current UX template.
<|end_header_id|>

<|start_header_id|>assistant<|end_header_id|>
Requirement 1: Add support for uip / uia import
Requirement 2: User must be able to import UIA and UIP files
<|end_header_id|>

### Example 3:
Backlog Item:
<|start_header_id|>user<|end_header_id|>
Summary: Support for qmlproject and Qt for MCUs
Description: Users should be able to create Qt for MCUs specific projects with Qt Design Studio. Qt Design Studio has its own project file format (.qmlproject). This project format has to support Qt for MCUs.
<|end_header_id|>

<|start_header_id|>assistant<|end_header_id|>
Requirement 1: Support for qmlproject and Qt for MCUs
Requirement 2: Users should be able to create Qt for MCUs specific projects with Qt Design Studio.
<|end_header_id|>

### Example 4:
Backlog Item:
<|start_header_id|>user<|end_header_id|>
Summary: Indicate if Property is Animated
Description: It would be helpful to indicate in the properties pane if a property is animated. Presently, if a binding is set the nut icon turns into a red gear, which is useful, so something like that but to indicate there's at least one keyframe set would be nice.
<|end_header_id|>

<|start_header_id|>assistant<|end_header_id|>
Requirement 1: Indicate if Property is Animated
Requirement 2: It would be helpful to indicate in the properties pane if a property is animated.
<|end_header_id|>

### Example 5:
Backlog Item:
<|start_header_id|>user<|end_header_id|>
Summary: Deployment to Qt for MCUs
Description: *Designers want to deploy Qt for MCUs based applications to the target hardware.* Ideally Qt Design Studio can deploy to devices. Currently, QDB is used for deployment. It is not very likely that Qt for MCUs can support the same deployment mechanism. Qt for MCUs requires compilation of a binary. Therefore we might fall back to Qt Creator for deployment. Having an easy way to deploy an application for a designer would be a key feature nevertheless.
<|end_header_id|>

<|start_header_id|>assistant<|end_header_id|>
Requirement 1: Having an easy way to deploy an application for a designer would be a key feature nevertheless.
Requirement 2: Designers want to deploy Qt for MCUs based applications to the target hardware.
Requirement 3: Deployment to Qt for MCUs.
<|end_header_id|>

"""

# Few-shot example requirements to exclude
few_shot_requirements = {
    "Gizmo design for rotate",
    "Add support for uip / uia import",
    "User must be able to import UIA and UIP files",
    "Support for qmlproject and Qt for MCUs",
    "Users should be able to create Qt for MCUs specific projects with Qt Design Studio.",
    "Indicate if Property is Animated",
    "It would be helpful to indicate in the properties pane if a property is animated.",
    "Having an easy way to deploy an application for a designer would be a key feature nevertheless.",
    "Designers want to deploy Qt for MCUs based applications to the target hardware.",
    "Deployment to Qt for MCUs."
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
    #("/storage/scratch/6538142/Network_Observability.xlsx", "/storage/scratch/6538142/network_observability1.json"),
    #("/storage/scratch/6538142/OpenShift_UX_Product_Design.xlsx", "/storage/scratch/6538142/openshift1.json"),
    ("/storage/scratch/6538142/Qt_Design_Studio.xlsx", "/storage/scratch/6538142/qtdesign21.json"),
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
