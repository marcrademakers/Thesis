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
Summary: Integrate cloud sources in OCM
Description: Goal: Using the user settings new feature (cloud sources) as a provider connection within the OCM flow. From the source's perspective: introduce OSD as a new app available for existing cloud source types. OCM: define a Source/Provider, then select the stored Source/Provider when creating a cluster. Requirements: * Allow adding OCM as an application using sources (for Azure, GCP, and AWS sources) * Provide the needed details according to the connection type (ARNs, IDs, etc.) * Support OSD/ ROSA/ ARO * Provide users the option to select an existing cloud source * Allow creating/ adding new sources from the OCM flow. Deliverables: Marvel mockups with different perspectives and variations or a Design doc * Initial Sources mocks: [https://marvelapp.com/prototype/74bc2hd] * Design doc and UX discussion [https://docs.google.com/document/d/1LYB5wjE_kNuVFLpFomh2NdcMnyEdmSQeSEAlcQ0AD-o/edit#heading=h.atlkp9j50fo7]. Stakeholders approval.
<|end_header_id|>

<|start_header_id|>assistant<|end_header_id|>
Requirement 1: Using the user settings new feature (cloud sources) as a provider connection within the OCM flow.
Requirement 2: Integrate cloud sources in OCM
<|end_header_id|>

### Example 2:
Backlog Item:
<|start_header_id|>user<|end_header_id|>
Summary: Surface ACM "Hubs" and Hybrid Console URLs within OCM
Description: OCM currently does not distinguish between "hub" clusters with ACM installed, "spoke" clusters that an ACM hub manages, or unattached clusters that don't have any connection to an ACM hub yet. As we start to encourage users to use ACM and Hybrid Console Hub UIs instead of single-cluster OpenShift Web Consoles pets, the OCM experience should be enhanced to point users to the appropriate Hybrid Console whenever possible. Success criteria: - In the Clusters List, somehow represent "Hub" clusters that have ACM installed - Consider where we might show the related "spoke" clusters being managed by the ACM hub as well - Consider what non-hub and non-spoke clusters should look like - OCM should point users trying to view a spoke cluster's UI to the related Hybrid Console - should we also show the OCP Web Console as an option, or not?
<|end_header_id|>

<|start_header_id|>assistant<|end_header_id|>
Requirement 1: As we start to encourage users to use ACM and Hybrid Console Hub UIs instead of single-cluster OpenShift Web Consoles pets, the OCM experience should be enhanced to point users to the appropriate Hybrid Console whenever possible.
<|end_header_id|>

### Example 3:
Backlog Item:
<|start_header_id|>user<|end_header_id|>
Summary: Add host using BMC - make provisioning network optional
Description: In case of adding a host via BMC, the infra will try to provision the host first using virtual media (for which the provisioning network is not needed) and if that is not possible (e.g. the HW does not support it), then it tries to use PXE booting. For PXE booting you need to have the provisioning network configured. So, the screen needs to change: 1: the provisioning network has to be optional 2: there needs to be an explanation that if it is not configured AND the boot via virtual media is not possible, the host provisioning will fail. We also need on the host list screen a state for host when the provisioning network is not configured and the boot using virtual media failed which will warn me that this happened and I can fix it by adding a provisioning network and then the host will be PXE booted (similar to the "add credentials" if the host has been discovered using discovery ISO so the BMC credentials are not there).
<|end_header_id|>

<|start_header_id|>assistant<|end_header_id|>
Requirement 1: Add host using BMC - make provisioning network optional
Requirement 2: The provisioning network has to be optional
<|end_header_id|>

### Example 4:
Backlog Item:
<|start_header_id|>user<|end_header_id|>
Summary: Align downloads abilities with installation pages
Description: Goal: Allow users to choose the architecture type not only from the Downloads page but also from the Installation pages for consistency and flexibility. Background: On the new Downloads page, we give access to a full set of OS/architecture choices. But on Create cluster pages, we only offer OS choice, assuming the user wants the same architecture as the cluster will run on — e.g. if they're [installing on Power|https://cloud.redhat.com/openshift/install/power/user-provisioned], they'll also get Linux installer and {{oc}} binaries for Power, while it's likely the user wants x86 instead. Similarly, the [pre-release page|https://cloud.redhat.com/openshift/install/pre-release] only offers x86 now, but other architectures exist on the server e.g. [https://mirror.openshift.com/pub/openshift-v4/s390x/clients/ocp-dev-preview/pre-release/]. Deliverables: Mockups or a design doc. Stakeholder acceptance.
<|end_header_id|>

<|start_header_id|>assistant<|end_header_id|>
Requirement 1: Align downloads abilities with installation pages
Requirement 2: Allow users to choose the architecture type not only from the Downloads page but also from the Installation pages for consistency and flexibility.
<|end_header_id|>

### Example 5:
Backlog Item:
<|start_header_id|>user<|end_header_id|>
Summary: [OCM] Final Designs: Improve the Transfer Ownership Workflow
Description: *Goal* * Transferring cluster ownership allows another individual to manage the cluster. Ownership can be transferred to other users in the same or different organization. [Link to documentation|https://access.redhat.com/solutions/4661621']. * The current workflow is not conducive to users making a connection between their evaluation expiring and the action of subscribing. Our goal is to encourage users to transfer their clusters to an existing account and link that directly to the evaluation expiration. *Background* * There are 1981 unclassified clusters. Having clusters in an unclassified state means that Red Hat does not understand what Custom Global Customer Name (GCN) the cluster is associated with. This limits Red Hat’s ability to engage with the owner of the cluster in a meaningful way which decreases chances of revenue generation, retention, and customer success. * There are several scenarios by which these unclassified clusters can occur. ([see User's image attached below or slide 5 in the deck|https://docs.google.com/presentation/d/1GJMATzWvLO7UA91dg49lGRXN6K2hxyu5bCHDPCbm97I/edit#slide=id.gf11562dcef_0_2]) * Low res wireframes have been approved from a strategic direction. We need to leverage those wireframes and create final designs. ([see slide 16 in the deck|https://docs.google.com/presentation/d/1GJMATzWvLO7UA91dg49lGRXN6K2hxyu5bCHDPCbm97I/edit#slide=id.gf11562dcef_0_47]) ** There was a workflow Gina had designed for when we know your organization. Perhaps we should consider that use case as well and try to include that scenario. Jake Lucky will have more background on why that may be difficult to do from an implementation standpoint. * Colleen/Megan can help with any additional background. *Requirements* * Leverage the eval status both on the clusters list and the cluster details to send users directly into a subscribe, transfer or buy flow: ** If the cluster is connected to an account and has an active subscription we will default to the Subscribe option ** If the clusters is not connected to an organization we will default to the transfer ownership and disable the subscription option * For the transfer ownership workflow: ** Improve the existing out of product \"Pull secret\" experience and bring as much of that into this modal/wizard as possible, while reducing the steps and making the UX as easy as possible for users. ** Add necessary cluster information to provide context for the user and to decrease the likely hood they transfer the wrong cluster ** Refine and understand the implementation necessary for an \"Email\" approach as an additional alternative to the \"Pull secret\" ** Consider adding a nudge to users about contacting their Org admin to be added to their existing Organizations's existing account if applicable. ** Consider what links to documentation would be necessary and useful. ** Identify any other gaps that may not have been accounted for. *** Do disconnected clusters need anything different? *Resources* * Slide deck: https://docs.google.com/presentation/d/1GJMATzWvLO7UA91dg49lGRXN6K2hxyu5bCHDPCbm97I/edit#slide=id.ge8ef82f34f_0_0 * Sketch files: https://sketch.com/s/3da04dc4-b859-42ca-96f5-85629d0b84b1 * Marvel: https://marvelapp.com/prototype/a77a24d/screen/82545002
<|end_header_id|>

<|start_header_id|>assistant<|end_header_id|>
Requirement 1: * Leverage the eval status both on the clusters list and the cluster details to send users directly into a subscribe, transfer or buy flow:
Requirement 2: If the cluster is connected to an account and has an active subscription we will default to the Subscribe option
Requirement 3: If the clusters is not connected to an organization we will default to the transfer ownership and disable the subscription option
Requirement 4: Add necessary cluster information to provide context for the user and to decrease the likely hood they transfer the wrong cluster
Requirement 5: Our goal is to encourage users to transfer their clusters to an existing account and link that directly to the evaluation expiration.
<|end_header_id|>

"""

# Few-shot example requirements to exclude
few_shot_requirements = {
    "Using the user settings new feature (cloud sources) as a provider connection within the OCM flow.",
    "Integrate cloud sources in OCM",
    "As we start to encourage users to use ACM and Hybrid Console Hub UIs instead of single-cluster OpenShift Web Consoles pets, the OCM experience should be enhanced to point users to the appropriate Hybrid Console whenever possible.",
    "Add host using BMC - make provisioning network optional",
    "The provisioning network has to be optional",
    "Align downloads abilities with installation pages",
    "Allow users to choose the architecture type not only from the Downloads page but also from the Installation pages for consistency and flexibility.",
    "* Leverage the eval status both on the clusters list and the cluster details to send users directly into a subscribe, transfer or buy flow:",
    "If the cluster is connected to an account and has an active subscription we will default to the Subscribe option",
    "If the clusters is not connected to an organization we will default to the transfer ownership and disable the subscription option",
    "Add necessary cluster information to provide context for the user and to decrease the likely hood they transfer the wrong cluster",
    "Our goal is to encourage users to transfer their clusters to an existing account and link that directly to the evaluation expiration."

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
    ("/storage/scratch/6538142/OpenShift_UX_Product_Design.xlsx", "/storage/scratch/6538142/openshift21.json"),
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
