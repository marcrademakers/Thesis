import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

# Load the aggregated results
results_file = "/Users/marcrademakers/Desktop/RQ2/RQ2.json"
with open(results_file, "r") as file:
    results = json.load(file)

# Create an output directory for saving plots
output_dir = "/Users/marcrademakers/Desktop/RQ2"
os.makedirs(output_dir, exist_ok=True)

# Extract data for visualizations
categories = list(results.keys())
original_counts = [data["general_statistics"]["total_references_original"] for data in results.values()]
new_counts = [data["general_statistics"]["total_references_new"] for data in results.values()]
overlap_ratios = [data["differences"]["average_overlap_ratio"] for data in results.values()]

unique_original = sum(data["differences"]["total_references_only_in_original"] for data in results.values())
unique_new = sum(data["differences"]["total_references_only_in_new"] for data in results.values())
overlap_total = sum(data["general_statistics"]["total_references_original"] for data in results.values()) - unique_original

# 1. Bar Chart: Total References per Granularity
def plot_total_references():
    x = range(len(categories))
    plt.bar(x, original_counts, width=0.4, label="Original", align="center")
    plt.bar([p + 0.4 for p in x], new_counts, width=0.4, label="New", align="center")
    plt.xticks([p + 0.2 for p in x], categories, rotation=45, ha="right")
    plt.ylabel("Total References")
    plt.title("Total References per Granularity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "total_references_per_granularity.png"))
    plt.show()

# 2. Pie Chart: Proportion of Differences
def plot_proportion_of_differences():
    labels = ["Unique to Original", "Unique to New", "Overlap"]
    sizes = [unique_original, unique_new, overlap_total]
    plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140, colors=["#ff9999", "#66b3ff", "#99ff99"])
    plt.title("Proportion of Differences")
    plt.savefig(os.path.join(output_dir, "proportion_of_differences.png"))
    plt.show()

# 3. Heatmap: Overlap Ratios
def plot_overlap_ratios():
    heatmap_data = [overlap_ratios]  # Single-row heatmap
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", xticklabels=categories, yticklabels=["Overlap Ratio"], cmap="YlGnBu")
    plt.title("Overlap Ratios by Category")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "overlap_ratios_heatmap.png"))
    plt.show()

# 4. Histogram: Distribution of Overlap Ratios
def plot_overlap_ratios_distribution():
    plt.hist(overlap_ratios, bins=10, color="skyblue", edgecolor="black")
    plt.xlabel("Overlap Ratio")
    plt.ylabel("Frequency")
    plt.title("Distribution of Overlap Ratios")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "overlap_ratios_distribution.png"))
    plt.show()

# Execute all plots
if __name__ == "__main__":
    print(f"Saving plots to {output_dir}...")
    plot_total_references()
    plot_proportion_of_differences()
    plot_overlap_ratios()
    plot_overlap_ratios_distribution()
    print("All plots saved successfully.")
