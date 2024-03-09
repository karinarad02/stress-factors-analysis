import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Load data from the CSV file
df = pd.read_csv('StressLevelDataset.csv')

# Display the first few rows of the data
print(df.head())

# Data cleaning
# Fill missing values with the mean for all columns
df = df.fillna(df.mean())

# Data standardization
# Calculate mean and standard deviation
mean_values = df.mean()
std_dev_values = df.std()
# Apply the standardization formula to the dataset
df_standard = (df - mean_values) / std_dev_values
print(df_standard)

# Initialize the PCA model
pca = PCA()

# Apply PCA to the standardized data
principal_components = pca.fit_transform(df_standard)

# Display the explained variance ratio for each principal component
explained_variance_ratio = pca.explained_variance_ratio_
print("Explained Variance Ratio:")
print(explained_variance_ratio)

# Factors with the biggest impact on components:
# Component 1: social_support
# Component 2: blood_pressure
principal_components_vectors = pca.components_
print("Eigenvectors associated with principal components:")
print(principal_components_vectors)

# Plot cumulative variance
plt.plot(np.cumsum(explained_variance_ratio))
plt.title('Cumulative Variance Plot')
plt.xlabel('Number of Principal Components')
plt.ylabel('Percentage of Explained Variance')
plt.show()

# Select principal components
num_components_to_keep = 2
principal_components_selected = principal_components[:, :num_components_to_keep]
print(principal_components_selected)

# Scatter plot
plt.scatter(
    principal_components_selected[:, 0], principal_components_selected[:, 1],
    c=df['social_support'], cmap='viridis', alpha=0.7
)
plt.colorbar(label='Anxiety Intensity based on Self-Esteem')
plt.title('Scatter Plot in Principal Components')
plt.xlabel('Principal Component 1 (social_support)')
plt.ylabel('Principal Component 2 (blood_pressure)')
plt.show()

# Box plot for Principal Component 1
plt.figure(figsize=(10, 6))
sns.boxplot(y=principal_components_selected[:, 0], data=df_standard)
plt.title('Box Plot for Principal Component 1')
plt.show()

# Box plot for Principal Component 2
plt.figure(figsize=(10, 6))
sns.boxplot(y=principal_components_selected[:, 1], data=df_standard)
plt.title('Box Plot for Principal Component 2')
plt.show()

# Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df_standard.corr(), cmap='coolwarm', annot=True, fmt=".2f", linewidths=.5,
            xticklabels=df.columns, yticklabels=df.columns,
            cbar_kws={'fraction': 0.05, 'pad': 0.05})
plt.title('Heatmap')
plt.show()
