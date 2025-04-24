# PLP-PYTHON-WEEK7assihnment

# ============================================
# Filename: data_analysis_with_pandas.py
# Description: This script loads and analyzes
# the Iris dataset using pandas and seaborn,
# and visualizes the data using matplotlib.
# Suitable for assignment submission.
# ============================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set Seaborn style for prettier plots
sns.set(style="whitegrid")

# -------------------------------
# Task 1: Load and Explore Dataset
# -------------------------------

print("=== Task 1: Loading and Exploring Dataset ===\n")

try:
    # Load the built-in Iris dataset from seaborn
    df = sns.load_dataset('iris')
    print("Dataset loaded successfully.\n")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# Display the first 5 rows of the dataset
print("First 5 rows of the dataset:")
print(df.head())

# Show the structure and types of the dataset
print("\nDataset Info:")
print(df.info())

# Check for missing values in each column
print("\nMissing values:")
print(df.isnull().sum())

# Clean data: Drop rows with missing values (if any)
df = df.dropna()

# -------------------------------
# Task 2: Basic Data Analysis
# -------------------------------

print("\n=== Task 2: Basic Data Analysis ===\n")

# Display descriptive statistics for numerical columns
print("Summary Statistics:")
print(df.describe())

# Group data by 'species' and calculate the mean for each group
grouped = df.groupby("species").mean()
print("\nMean values grouped by species:")
print(grouped)

# Key observation from analysis
print("\nObservation: Iris-virginica has the highest average petal length.\n")

# -------------------------------
# Task 3: Data Visualization
# -------------------------------

print("=== Task 3: Visualizing the Data ===\n")

# 1. Line Chart - Sepal length across samples
plt.figure(figsize=(8, 4))
plt.plot(df.index, df["sepal_length"], label="Sepal Length", color='purple')
plt.title("Sepal Length Trend")
plt.xlabel("Index (Sample Number)")
plt.ylabel("Sepal Length (cm)")
plt.legend()
plt.tight_layout()
plt.savefig("line_chart.png")
plt.show()

# 2. Bar Chart - Average petal length per species
plt.figure(figsize=(8, 5))
sns.barplot(x=grouped.index, y=grouped["petal_length"], palette="Set2")
plt.title("Average Petal Length per Species")
plt.xlabel("Species")
plt.ylabel("Petal Length (cm)")
plt.tight_layout()
plt.savefig("bar_chart.png")
plt.show()

# 3. Histogram - Distribution of Sepal Width
plt.figure(figsize=(8, 5))
plt.hist(df["sepal_width"], bins=15, color='skyblue', edgecolor='black')
plt.title("Distribution of Sepal Width")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("histogram.png")
plt.show()

# 4. Scatter Plot - Sepal Length vs. Petal Length
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="sepal_length", y="petal_length", hue="species", palette="Set1")
plt.title("Sepal Length vs. Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.tight_layout()
plt.savefig("scatter_plot.png")
plt.show()

# -------------------------------
# End of Script
# -------------------------------

print("Analysis and visualizations completed. Plots saved as images.")
