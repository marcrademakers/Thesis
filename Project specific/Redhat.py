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
Summary: Add caption to video on Product Overview page
Description: I'd like to add a simple description to the video that is added to a Product's Overview tab. This should be an optional field, not required one. h2. Stakeholders Jason Porter [~mguerett] h2. Acceptance Criteria {{noformat}} Given that there is a product page in Drupal And I am able to edit that product Then a new field is available for a video caption/description on the overview page And when that field is populated and saved the caption displays beneath the video on the overview page {{noformat}} h2. Review Steps * Modify an existing product * Add a caption for the video * Verify that caption displays beneath the view on the overview page
<|end_header_id|>

<|start_header_id|>assistant<|end_header_id|>
Requirement 1: Add caption to video on Product Overview page
Requirement 2: I'd like to add a simple description to the video that is added to a Product's Overview tab.
Requirement 3: This should be an optional field, not required one.
Requirement 4: Given that there is a product page in Drupal And I am able to edit that product Then a new field is available for a video caption/description on the overview page And when that field is populated and saved the caption displays beneath the video on the overview page
<|end_header_id|>

### Example 2:
Backlog Item:
<|start_header_id|>user<|end_header_id|>
Summary: Add support for OS detection for the pervasive red DOWNLOAD buttons
Description: h2. Requirements We need to be able to detect the user's OS for the CDK and Red Hat Development Suite products and change the primary CTA for these products on both their product-specific Download page and the site Downloads page. Refer to [OS based downloads in the Requirements Doc |https://docs.google.com/document/d/1m1WljXUSM8EonxaFKtDTmysxrWUnssH1PiShKZ6Mz1o/edit?ts=59c0d31b#heading=h.i6hwqauxuh05] for full details Other Notes from Nick: So that the pervasive red "DOWNLOAD" button can link to the correct download for the user's platform, we need some kind of javascript based OS detection. Today the button just links to the FIRST available download on the download page [4], [5], regardless of OS. Then, rather than a Windows user being prompted to download the linux version of the CDK or the Developer Suite installer, they'd get the Windows-appropriate [1] one. Same story for MacOS [2]. But for Linux users, we'd need to redirect them to another page [3] to explain how to enable the software collection, import the signing key, and install via yum. [1] https://developers.redhat.com/download-manager/file/devsuite-2.0.0-GA-installer.exe [2] https://developers.redhat.com/download-manager/file/devsuite-2.0.0-GA-installer-mac.zip [3] https://developers.redhat.com/products/devsuite/hello-world/#fndtn-rhel See also https://projects.engineering.redhat.com/browse/RCM-21427, where the request to add the devsuite installer binaries to the CDK downloads page is being discussed. h2. Design Invision Wireframes: https://redhat.invisionapp.com/share/DBDLRYBN8#/254627258_Product_Download_Page h2. Stakeholders UX: [~gdoyle-1] ENG: [~dcoughlin1]
<|end_header_id|>

<|start_header_id|>assistant<|end_header_id|>
Requirement 1: Add support for OS detection for the pervasive red DOWNLOAD buttons
Requirement 2: We need to be able to detect the user's OS for the CDK and Red Hat Development Suite products and change the primary CTA for these products on both their product-specific Download page and the site Downloads page.
<|end_header_id|>

### Example 3:
Backlog Item:
<|start_header_id|>user<|end_header_id|>
Summary: Add FileURL for DTM Download Tracking
Description: See this doc for more information https://docs.google.com/a/redhat.com/document/d/1QgO7_uqSwQM-GpmHglZqtcu42g9YmykvVxlc0PuCeEY/edit?disco=AAAABPHecCc h2. Requirements Marcia has requested a new data point be added to download data: fileURL h2. Design * NA h2. Stakeholders * Engineering-only h2. Acceptance Criteria * fileURL data point is included where applicable in DTM download event h2. Review Steps * download a file from the site * note the Analytics event data should include a File URL
<|end_header_id|>

<|start_header_id|>assistant<|end_header_id|>
Requirement 1: Add FileURL for DTM Download Tracking
<|end_header_id|>

### Example 4:
Backlog Item:
<|start_header_id|>user<|end_header_id|>
Summary: Convert <h1> tags in archives to <h2>
Description: Following discussions with Jairus on SEO, we learned that there should be only one <h1> heading in the HTML on any given page. The archive pages – categories, author page, tag pages and front page, all have multiple <h1> enclosed post titles/links. On the front page, the first post should be the only <h1> (unless a permanent title is added later). On the category, tag and author page, all post titles should be <h2>, since the pages all already have one <h1> heading.
<|end_header_id|>

<|start_header_id|>assistant<|end_header_id|>
Requirement 1: Convert <h1> tags in archives to <h2>
Requirement 2: On the front page, the first post should be the only <h1> (unless a permanent title is added later).
Requirement 3: On the category, tag and author page, all post titles should be <h2>, since the pages all already have one <h1> heading.
<|end_header_id|>

### Example 5:
Backlog Item:
<|start_header_id|>user<|end_header_id|>
Summary: Create sortable table component
Description: h2. Requirements [~mguerett] has requested a way to create a sortable table of data for content. We'll need to add a custom element that will be a Red Hat approved tag that can be used to denote a sortable table. h2. Design * Engineering-only (with existing styles) * [Invision Wireframes|https://redhat.invisionapp.com/share/WRE1LVQT4#/259322893_Sortable_Table] h2. Stakeholders * Approval - [~mguerett] * Eng - [~ldary24] * UX Consult - [~gdoyle-1] h2. Acceptance Criteria * As a content creator I should be able to create a table that is sortable by the column headings * If I surround a table element with the <rh-sortable> tag it should automatically enable sorting the table by the column headers h3. Feature: Sortable Table h3. Scenario 1 Given I am on a page with a sortable table of data Then display two carets on top of each other And display them to the right of the column header. h3. Scenario 2 Given I am on a page with a sortable table of data When I select a column header Display a blue bar above the heading of the column And display the column header and top caret to the right of the heading in blue And sort the data from top to bottom. h3. Scenario 3 Given I have selected a column and the elements of Scenario 2 have been met When I select the same column header again Display the bottom caret to the right of the heading in blue And sort the data from bottom to top. h3. Scenario 4 Given I have selected a column and elements of Scenario 3 have been met When I select the same column header again Display the top caret to the right of the heading in blue And sort the data from top to bottom (continue this cycle if the column header is selected again). h3. Scenario 5 Given I have selected a column and the elements of Scenario 2 have been met When I select a different column header Display two carets on top of each other to the right of all column headers other than the selected one. h2. Review Steps * Create a Basic Page * Put a table element in using column headers and at least a couple rows of data * Surround the table element with the <rh-sortable> </rh-sortable> tag * Save the page and view it * The column headers should sort the data in a sensible fashion (text versus numbers).
<|end_header_id|>

<|start_header_id|>assistant<|end_header_id|>
Requirement 1: As a content creator I should be able to create a table that is sortable by the column headings
Requirement 2: If I surround a table element with the <rh-sortable> tag it should automatically enable sorting the table by the column headers
Requirement 3: Given I am on a page with a sortable table of data Then display two carets on top of each other And display them to the right of the column header.
Requirement 4: Given I am on a page with a sortable table of data When I select a column header Display a blue bar above the heading of the column And display the column header and top caret to the right of the heading in blue And sort the data from top to bottom.
Requirement 5: Given I have selected a column and the elements of Scenario 2 have been met When I select the same column header again Display the bottom caret to the right of the heading in blue And sort the data from bottom to top.
Requirement 6: Given I have selected a column and elements of Scenario 3 have been met When I select the same column header again Display the top caret to the right of the heading in blue And sort the data from top to bottom (continue this cycle if the column header is selected again).
Requirement 7: Given I have selected a column and the elements of Scenario 2 have been met When I select a different column header Display two carets on top of each other to the right of all column headers other than the selected one.
Requirement 8: We'll need to add a custom element that will be a Red Hat approved tag that can be used to denote a sortable table.
Requirement 9: Create sortable table component.
<|end_header_id|>

"""

# Few-shot example requirements to exclude 
few_shot_requirements = {
    "Add caption to video on Product Overview page",
    "I'd like to add a simple description to the video that is added to a Product's Overview tab.",
    "This should be an optional field, not required one.",
    "Given that there is a product page in Drupal And I am able to edit that product Then a new field is available for a video caption/description on the overview page And when that field is populated and saved the caption displays beneath the video on the overview page",
    "Add support for OS detection for the pervasive red DOWNLOAD buttons",
    "We need to be able to detect the user's OS for the CDK and Red Hat Development Suite products and change the primary CTA for these products on both their product-specific Download page and the site Downloads page.",
    "Add FileURL for DTM Download Tracking",
    "Convert <h1> tags in archives to <h2>",
    "On the front page, the first post should be the only <h1> (unless a permanent title is added later).",
    "On the category, tag and author page, all post titles should be <h2>, since the pages all already have one <h1> heading.",
    "As a content creator I should be able to create a table that is sortable by the column headings",
    "If I surround a table element with the <rh-sortable> tag it should automatically enable sorting the table by the column headers",
    "Given I am on a page with a sortable table of data Then display two carets on top of each other And display them to the right of the column header.",
    "Given I am on a page with a sortable table of data When I select a column header Display a blue bar above the heading of the column And display the column header and top caret to the right of the heading in blue And sort the data from top to bottom.",
    "Given I have selected a column and the elements of Scenario 2 have been met When I select the same column header again Display the bottom caret to the right of the heading in blue And sort the data from bottom to top.",
    "Given I have selected a column and elements of Scenario 3 have been met When I select the same column header again Display the top caret to the right of the heading in blue And sort the data from top to bottom (continue this cycle if the column header is selected again).",
    "Given I have selected a column and the elements of Scenario 2 have been met When I select a different column header Display two carets on top of each other to the right of all column headers other than the selected one.",
    "We'll need to add a custom element that will be a Red Hat approved tag that can be used to denote a sortable table.",
    "Create sortable table component."

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
    #("/storage/scratch/6538142/Qt_Design_Studio.xlsx", "/storage/scratch/6538142/qtdesign1.json"),
    ("/storage/scratch/6538142/Red_Hat_Developer_Website_v2.xlsx", "/storage/scratch/6538142/redhat21.json")
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
