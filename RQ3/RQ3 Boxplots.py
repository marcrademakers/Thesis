import numpy as np
import matplotlib.pyplot as plt

# Define the data for each setup
zero_shot = np.array([
    [0.206, 0.255, 0.228],  # Jira
    [0.531, 0.508, 0.519],  # Lyrasis
    [0.326, 0.433, 0.372],  # Network Obs
    [0.360, 0.412, 0.384],  # OpenShift
    [0.592, 0.381, 0.464],  # Qt Design
    [0.354, 0.460, 0.400]   # Red Hat
])

cross_project = np.array([
    [0.297, 0.733, 0.423],  # Jira
    [0.492, 0.733, 0.589],  # Lyrasis
    [0.330, 0.678, 0.444],  # Network Obs
    [0.348, 0.628, 0.448],  # OpenShift
    [0.383, 0.579, 0.461],  # Qt Design
    [0.271, 0.406, 0.325]   # Red Hat
])

project_specific = np.array([
    [0.274, 0.537, 0.363],  # Jira
    [0.483, 0.311, 0.378],  # Lyrasis
    [0.403, 0.651, 0.498],  # Network Obs
    [0.528, 0.704, 0.603],  # OpenShift
    [0.489, 0.657, 0.561],  # Qt Design
    [0.419, 0.617, 0.499]   # Red Hat
])

# Define labels for the setups
setup_labels = ["Zero-shot", "Cross-project Few-shot", "Project-specific Few-shot"]

# Define metric names
metrics = ["Precision", "Recall", "F1-score"]

# Store the data in a list for easy iteration
data = [zero_shot, cross_project, project_specific]

# Create box plots for each metric
for i, metric in enumerate(metrics):
    plt.figure(figsize=(8, 6))
    plt.boxplot([setup[:, i] for setup in data], labels=setup_labels, patch_artist=True)
    plt.title(f"Box Plot of {metric} Across Setups")
    plt.ylabel(metric)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()
