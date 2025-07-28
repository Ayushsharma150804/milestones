#Matplotlib and Seaborn.
#mall_customets

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


df = pd.read_csv("Mall_Customers.csv")


print(" First 5 Rows:")
print(df.head())


print("\n Dataset Info:")
print(df.info())


print("\n Missing Values:")
print(df.isnull().sum())


print("\n Statistical Summary:")
print(df.describe())


df.columns = ['CustomerID', 'Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']


plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Gender', hue='Gender', palette='pastel', legend=False)
plt.title("Gender Distribution")
plt.show()


plt.figure(figsize=(8, 5))
sns.histplot(df['Age'], bins=20, kde=True, color='skyblue')
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()


plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Gender')
plt.title("Income vs Spending Score")
plt.show()


sns.pairplot(df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']], diag_kind='kde')
plt.suptitle("Pairwise Plots", y=1.02)
plt.show()


plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='Gender', y='Age', hue='Gender', palette='Set2', dodge=False, legend=False)
plt.title("Age Distribution by Gender")
plt.show()


plt.figure(figsize=(6, 4))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

#salary_data

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


df = pd.read_csv("Salary_Data.csv")


print(" First 5 Rows:")
print(df.head())


print("\n Dataset Info:")
print(df.info())


print("\n Missing Values:")
print(df.isnull().sum())


print("\n Statistical Summary:")
print(df.describe())

plt.figure(figsize=(6, 4))
sns.histplot(df['YearsExperience'], bins=10, kde=True, color='lightblue')
plt.title("Years of Experience Distribution")
plt.xlabel("Years of Experience")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(6, 4))
sns.histplot(df['Salary'], bins=10, kde=True, color='salmon')
plt.title("Salary Distribution")
plt.xlabel("Salary")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(7, 5))
sns.scatterplot(data=df, x='YearsExperience', y='Salary', color='green')
plt.title("Salary vs Years of Experience")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.grid(True)
plt.show()

plt.figure(figsize=(5, 4))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()


#Social Network Ads

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


warnings.simplefilter(action='ignore', category=FutureWarning)


df = pd.read_csv("Social_Network_Ads.csv")


print(" First 5 Rows:")
print(df.head())


print("\n Dataset Info:")
print(df.info())


print("\n Missing Values:")
print(df.isnull().sum())


print("\n Statistical Summary:")
print(df.describe())


print("\n Purchase Distribution:")
print(df['Purchased'].value_counts())

plt.figure(figsize=(5, 4))
sns.countplot(data=df, x='Gender', palette='pastel')
plt.title("Gender Distribution")
plt.show()

plt.figure(figsize=(6, 4))
sns.histplot(data=df, x='Age', hue='Purchased', multiple='stack', palette='Set2', kde=True)
plt.title("Age vs Purchased")
plt.show()

plt.figure(figsize=(6, 4))
sns.histplot(data=df, x='EstimatedSalary', hue='Purchased', multiple='stack', palette='Set1', kde=True)
plt.title("Estimated Salary vs Purchased")
plt.show()

sns.pairplot(df, hue='Purchased', palette='husl')
plt.suptitle("Pairwise Relationships", y=1.02)
plt.show()

plt.figure(figsize=(6, 4))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
