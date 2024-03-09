import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Încarcă datele din fișierul CSV
df = pd.read_csv('StressLevelDataset1.csv')

# Afișează primele câteva rânduri ale datelor
print(df.head())

# Curățare date
# Umple valorile lipsă cu media pentru toate coloanele
df = df.fillna(df.mean())

# Standardizarea datelor
# Calcul medie și deviație standard
mean_values = df.mean()
std_dev_values = df.std()
# Aplicare formula de standardizare pe setul de date
df_standard = (df - mean_values) / std_dev_values
print(df_standard)

# Inițializare modelul ACP
pca = PCA()

# Aplicare ACP pe datele standardizate
principal_components = pca.fit_transform(df_standard)

# Afișare raport varianță explicativă pentru fiecare componentă principală
explained_variance_ratio = pca.explained_variance_ratio_
print("Raportul varianței explicative:")
print(explained_variance_ratio)

principal_components_vectors = pca.components_
print("Vectorii proprii asociați componentelor principale:")
print(principal_components_vectors)

# Plot varianță cumulativă
plt.plot(np.cumsum(explained_variance_ratio))
plt.title('Plot varianță cumulativă')
plt.xlabel('Numărul de componente principale')
plt.ylabel('Procentul de varianță explicat')
plt.show()

# Selectare componente principale
num_components_to_keep = 2
principal_components_selected = principal_components[:, :num_components_to_keep]
print(principal_components_selected)

# Scatter plot
plt.scatter(
    principal_components_selected[:, 0], principal_components_selected[:, 1],
    c=df['anxiety_level'], cmap='viridis', alpha=0.7
)
plt.colorbar(label='Intensitate anxietate în funcție de stima de sine')
plt.title('Scatter Plot în Componentele Principale')
plt.xlabel('Componenta Principală 1 (nivel de anxietate)')
plt.ylabel('Componenta Principală 2 (stima de sine)')
plt.show()

# Box plot pentru Componenta Principală 1
plt.figure(figsize=(10, 6))
sns.boxplot(y=principal_components_selected[:, 0], data=df_standard)
plt.title('Box Plot pentru Componenta Principală 1')
plt.show()

# Box plot pentru Componenta Principală 2
plt.figure(figsize=(10, 6))
sns.boxplot(y=principal_components_selected[:, 1], data=df_standard)
plt.title('Box Plot pentru Componenta Principală 2')
plt.show()

# Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df_standard.corr(), cmap='coolwarm', annot=True, fmt=".2f", linewidths=.5,xticklabels=df.columns, yticklabels=df.columns,cbar_kws={'fraction': 0.05, 'pad': 0.05})
plt.title('Heatmap')
plt.show()
