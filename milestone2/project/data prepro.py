import pandas as pd

df = pd.read_csv('melb_data.csv')

print("Initial Dataset Info:")
print(df.info())
print("\nFirst 5 rows:")
print(df.head())


if 'Address' in df.columns:
    df.drop(['Address'], axis=1, inplace=True)

print("\nMissing data columns:")
print(df.isnull().sum()[df.isnull().sum() > 0])

for col in df.columns:
    if df[col].dtype == 'object':
        df[col].fillna(df[col].mode()[0], inplace=True)  
    else:
        df[col].fillna(df[col].median(), inplace=True)

print("\nMissing data handled.")
print(df.isnull().sum().sum(), "missing values remain.")


df = pd.get_dummies(df, drop_first=True)

print("\nAfter encoding categorical variables:")
print(df.info())
