# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Optional: display plots inside Jupyter Notebook
# %matplotlib inline  

# -------------------------------
# Task 1: Load and Explore Dataset
# -------------------------------

# Load the dataset
try:
    # Example: Load Iris dataset from seaborn
    df = sns.load_dataset('iris')  # Replace with pd.read_csv('your_file.csv') for external CSV
    print("Dataset loaded successfully!\n")
except FileNotFoundError:
    print("Error: File not found.")
except Exception as e:
    print(f"An error occurred: {e}")

# Display first 5 rows
print("First 5 rows of the dataset:")
print(df.head())


# Dataset info and missing values
print("\nDataset Info:")
print(df.info())

print("\nMissing values in each column:")
print(df.isnull().sum())

# Clean missing values (if any)
df = df.dropna()

# -------------------------------
# Task 2: Basic Data Analysis
# -------------------------------

# Summary statistics
print("\nSummary Statistics:")
print(df.describe())

# Grouping by species and calculating mean
grouped = df.groupby('species').mean()
print("\nMean values by species:")
print(grouped)

# -------------------------------
# Task 3: Data Visualization
# -------------------------------

# Set Seaborn style
sns.set(style="whitegrid")

# Line Chart - Sample: trend over index (not time-series in Iris, just for illustration)
plt.figure(figsize=(8, 4))
plt.plot(df.index, df["sepal_length"], label="Sepal Length")
plt.title("Sepal Length Trend")
plt.xlabel("Index")
plt.ylabel("Sepal Length")
plt.legend()
plt.show()

# Bar Chart - Average petal length per species
plt.figure(figsize=(8, 5))
sns.barplot(x=grouped.index, y=grouped['petal_length'])
plt.title("Average Petal Length per Species")
plt.xlabel("Species")
plt.ylabel("Petal Length")
plt.show()

# Histogram - Distribution of sepal width
plt.figure(figsize=(8, 5))
plt.hist(df["sepal_width"], bins=15, color='skyblue', edgecolor='black')
plt.title("Distribution of Sepal Width")
plt.xlabel("Sepal Width")
plt.ylabel("Frequency")
plt.show()

# Scatter Plot - Sepal Length vs. Petal Length
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='sepal_length', y='petal_length', hue='species')
plt.title("Sepal Length vs. Petal Length")
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.legend()
plt.show()