import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the data
df = pd.read_csv("diabetes.csv")

# Display basic info
print("Dataset Info:")
print(df.info())
print("\nFirst 5 rows:")
print(df.head())

# 2. Data Pre-processing

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Replace zero values where they are not valid
cols_with_zero_invalid = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in cols_with_zero_invalid:
    df[col] = df[col].replace(0, df[col].median())  # ✅ Corrected chained assignment

print("\nZero-value columns replaced with median.")

# 3. Handle Categorical Data

# Outcome is categorical: 0 = Non-Diabetic, 1 = Diabetic
df['Outcome'] = df['Outcome'].map({0: 'Non-Diabetic', 1: 'Diabetic'})
print("\nConverted Outcome to categorical labels.")

# 4. Uni-variate Analysis

# Plot distribution for each numeric feature
for col in df.columns[:-1]:  # Exclude 'Outcome'
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col], kde=True, bins=30, color='skyblue')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Count plot for Outcome (Categorical)
plt.figure(figsize=(5, 4))
sns.countplot(x='Outcome', data=df, hue='Outcome', palette='Set2', legend=False)  # ✅ Fixed future warning
plt.title("Count of Diabetic vs Non-Diabetic")
plt.xlabel("Outcome")
plt.ylabel("Count")
plt.grid(True)
plt.tight_layout()
plt.show()
