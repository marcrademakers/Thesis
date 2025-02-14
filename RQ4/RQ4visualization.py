import numpy as np
import matplotlib.pyplot as plt

# Recall data for Requirement Type (NFR, User, System)
recall_type = np.array([
    [0.301653, 0.512397, 0.466942],  # NFR
    [0.324713, 0.406609, 0.416667],  # User
    [0.211180, 0.385093, 0.229814]   # System
]).T  # Transpose so rows = setups, columns = categories

# Recall data for Requirement Level (High, Medium, Low)
recall_level = np.array([
    [0.041096, 0.315068, 0.191781],  # High
    [0.300401, 0.500668, 0.409880],  # Medium
    [0.379061, 0.256318, 0.429603]   # Low
]).T  # Transpose so rows = setups, columns = categories

# Category labels
categories_type = ["Nfr", "User", "System"]
categories_level = ["High", "Medium", "Low"]

# Create the box plot for Requirement Type
plt.figure(figsize=(12, 6))
plt.boxplot(recall_type, labels=categories_type, patch_artist=True)
plt.title("Box Plot of Recall by Requirement Type (Nfr, User, System)")
plt.ylabel("Recall Score")
plt.xlabel("Requirement Type")
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Show the first plot
plt.show()

# Create the box plot for Requirement Level
plt.figure(figsize=(12, 6))
plt.boxplot(recall_level, labels=categories_level, patch_artist=True)
plt.title("Box Plot of Recall by Requirement Level (High, Medium, Low)")
plt.ylabel("Recall Score")
plt.xlabel("Requirement Level")
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Show the second plot
plt.show()
