import json

def remove_unwanted_requirements(json_file, output_file, requirements_to_remove):
    """
    Removes JSON entries where:
      1. The entry matches any entry in `requirements_to_remove` (case-insensitive, stripped of extra spaces).
      2. Supports deeply nested dictionary structures.

    :param json_file: Path to the input JSON file.
    :param output_file: Path to save the cleaned JSON.
    :param requirements_to_remove: A list of requirement strings to remove.
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total_removed = 0  # Counter for removed items

    # Normalize the list of requirements to remove (strip whitespace, lowercase)
    normalized_remove_list = {req.strip().lower() for req in requirements_to_remove}

    # Ensure JSON is a dictionary with nested dictionaries containing lists
    if isinstance(data, dict):
        for category, projects in data.items():
            if isinstance(projects, dict):  # Process project-level dictionaries
                for project_name, requirements in projects.items():
                    if isinstance(requirements, list):  # Process requirement lists
                        original_length = len(requirements)

                        # Remove unwanted requirements
                        data[category][project_name] = [
                            req for req in requirements if req.strip().lower() not in normalized_remove_list
                        ]

                        removed_count = original_length - len(data[category][project_name])
                        total_removed += removed_count

    else:
        raise ValueError("JSON file structure is not supported. Expected a nested dictionary with lists.")

    # Save the cleaned data
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"✅ Cleaned JSON saved to {output_file}")
    print(f"🗑️ Removed {total_removed} requirement(s).")


# Define the file paths
json_file = "/Users/marcrademakers/Desktop/RQ4/consolidated_data.json"
output_file = "/Users/marcrademakers/Desktop/RQ4/rq4groundtruth.json"

# List of requirements to remove (ALL references you provided)
requirements_to_remove = [


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
    "Send Data to S3",
    "Make legend for JPT charts more usable",
    "Customise stack creation timeouts",
    "Provide builders in infrastructure",
    "Install Jira distribution and download jira home in parallel",
    "Aws infrastructure could provide running instances faster if we download dataset and jira installer in parallel.",
    "I want to use `docker-infrastructure` in infrastructure's tests. (infrastructure depends on docker-infrastructure).",
    "I want to use `infrastructure` in `docker-infrastructure` the same way we use it in `aws-infrastructure` (Installing Java, Product etc.).",
    "Extract infrastructure integration tests to a sub-module.",
    "UI: Handle Glacier storage state errors gracefully",
    "This task is to provide users with useful feedback when they attempt any of these actions on a content item in Glacier. More specifically, when a call is made on a Glacier content item via the storeclient, a ContentStateException is thrown. (If the content item was recently added or has been retrieved from Glacier, it will simply download as expected) This task is to show the user an appropriate error message indicating that the content is in Glacier. For now, that error message can indicate that the user should contact DuraCloud support if they wish to retrieve that content item.",
    "This improvement is to create a dryrun flag on the synctool. This flag would tell the synctool to go through the steps of determining which files should be sync-ed up to DuraCloud (including deleted and modified content), but instead of actually performing the sync, just a report will be produced detailing which content would have been sync-ed.",
    "SystemUserCredential Across Webapps",
    "This improvement is to create the SystemUserCredential in a similar was to how RootUserCredential is created, via a default username/password over-written by system properties. Once this is done, DuradminExecutorImpl.java should be evaluated for exchanging root-user with system-user.",
    "This feature is to provide a means for users to encrypt/decrypt content stored and retrieved through the StoreClient.  The StoreClient should have a constructor that optionally takes a symmetric or asymmetric encryption key.",
    "StoreClient encryption support.",
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
    "UI: Smart column sorting.",
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
    "Our goal is to encourage users to transfer their clusters to an existing account and link that directly to the evaluation expiration.",
    "Gizmo design for rotate",
    "Add support for uip / uia import",
    "User must be able to import UIA and UIP files",
    "Support for qmlproject and Qt for MCUs",
    "Users should be able to create Qt for MCUs specific projects with Qt Design Studio.",
    "Indicate if Property is Animated",
    "It would be helpful to indicate in the properties pane if a property is animated.",
    "Having an easy way to deploy an application for a designer would be a key feature nevertheless.",
    "Designers want to deploy Qt for MCUs based applications to the target hardware.",
    "Deployment to Qt for MCUs.",
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
]

# Run the function to clean the JSON file
remove_unwanted_requirements(json_file, output_file, requirements_to_remove)
