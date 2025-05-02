
# Dataset analysis with PCA
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Load and Process the dataset
file_path = '.../Dataset.csv'  # Update the file path
data = pd.read_csv(file_path)
feature_columns = ['d-band center', 'd-band filling', 'd-band width', 'd-band upper edge']
features = data[feature_columns]
scaler = StandardScaler()
standardized_features = scaler.fit_transform(features)
pca = PCA()
pca.fit(standardized_features)
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance = explained_variance_ratio.cumsum()
def customize_plot(xlabel, ylabel, xticks=None, yticks=None, xlim=None, ylim=None):
    plt.xlabel(xlabel, fontsize=24, fontweight="bold")
    plt.ylabel(ylabel, fontsize=24, fontweight="bold")
    if xticks is not None:
        plt.xticks(xticks, fontsize=24, fontweight="bold")
    else:
        plt.xticks(fontsize=24, fontweight="bold")
    if yticks is not None:
        plt.yticks(yticks, fontsize=24, fontweight="bold")
    else:
        plt.yticks(fontsize=24, fontweight="bold")
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    for spine in ['top', 'right', 'left', 'bottom']:
        plt.gca().spines[spine].set_linewidth(4)
        plt.gca().spines[spine].set_color('black')
    plt.tight_layout()

# Plot 1: Variance explained by each PC (Bar Plot with Cumulative Variance)
plt.figure(figsize=(12, 8))
plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio * 100, alpha=0.7, color="blue", linewidth=4)
plt.xticks(range(1, len(explained_variance_ratio) + 1), fontsize=24, fontweight="bold")
customize_plot("Principal Component", "Variance Explained [%]", xlim=(0.5, len(explained_variance_ratio) + 0.5))
plt.show()

# Plot 2: Scree Plot (Variance Explained by Each PC)
plt.figure(figsize=(12, 8))
plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio * 100, marker='o', linestyle='-', color="blue", linewidth=4)
plt.xticks(range(1, len(explained_variance_ratio) + 1), fontsize=24, fontweight="bold")
customize_plot("Principal Component", "Variance Explained [%]", xlim=(0.5, len(explained_variance_ratio) + 0.5))
plt.show()

# Plot 3: Cumulative Variance Explained
plt.figure(figsize=(12, 8))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance * 100, marker='o', linestyle='--', color='red', linewidth=4)
plt.xticks(range(1, len(explained_variance_ratio) + 1), fontsize=24, fontweight="bold")
customize_plot("Number of Components", "Cumulative Variance [%]", xlim=(0.5, len(cumulative_variance) + 0.5))
plt.show()

# Plot 4: PCA Loadings Heatmap
loadings = pd.DataFrame(pca.components_, columns=feature_columns)
plt.figure(figsize=(10, 8))

# Create the heatmap and assign the color bar
heatmap = sns.heatmap(
    loadings,
    cmap='coolwarm',
    annot=False,
    cbar_kws={'label': 'Loading Value'}  # Add label to the color bar
)

# Access the color bar from the heatmap
cbar = heatmap.collections[0].colorbar
cbar.ax.set_ylabel('Loading Value', fontsize=24, fontweight="bold")  # Set label font size and weight
cbar.ax.tick_params(labelsize=24, width=2)  # Adjust tick font size and thickness

# Customize the tick labels
for label in cbar.ax.get_yticklabels():
    label.set_fontweight("bold")  # Make the tick labels bold

# Customize the heatmap axis
plt.yticks(range(len(loadings.index)), [f"{i+1}" for i in range(len(loadings.index))], fontsize=24, fontweight="bold")
plt.xticks(rotation=90, fontsize=24, fontweight="bold")  # Rotate x-axis labels vertically
customize_plot("Features", "Principal Component")
plt.show()

