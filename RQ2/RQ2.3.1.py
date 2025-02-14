import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define confusion matrix with appropriate labels
cm = np.array([[921, 181], 
               [254, 921]])

# Define labels for rows and columns
labels = ["Agreements", "Missing"]
columns = ["Removed", "Agreements"]

# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", linewidths=0.5, xticklabels=columns, yticklabels=labels)

plt.xlabel("Round 2")
plt.ylabel("Round 1")
plt.title("Confusion matrix agreements")

# Show plot
plt.show()
