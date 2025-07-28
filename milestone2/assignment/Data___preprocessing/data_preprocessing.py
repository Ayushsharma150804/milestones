#preprocessing using sk_learn

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


df = pd.read_csv("melb_data.csv")


print("🔹 First 5 rows:")
print(df.head())


print("\nStatistical Summary:")
print(df.describe(include='all'))


X = df.drop('Price', axis=1)
y = df['Price']


numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['object']).columns

print("\n Numeric columns:", list(numeric_cols))
print(" Categorical columns:", list(categorical_cols))


numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])


preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("\n Preprocessing Complete!")
print(" Shape of training data (after preprocessing):", X_train_processed.shape)
print(" Shape of test data:", X_test_processed.shape)

#data   preprocessing using pandas

import pandas as pd


df = pd.read_csv("melb_data.csv")


print("🔹 First 5 rows:")
print(df.head())


print("\n📐 Shape:", df.shape)
print("🧱 Columns:", df.columns.tolist())


print("\n❓ Missing Values:")
print(df.isnull().sum())


threshold = len(df) * 0.3
df = df.dropna(axis=1, thresh=threshold)


numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

categorical_cols = df.select_dtypes(include=['object']).columns
df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])


if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')


for col in categorical_cols:
    df[col] = df[col].astype('category').cat.codes


print("\n📊 Statistical Summary:")
print(df.describe())

print("\n🔗 Correlation Matrix:")
print(df.corr())

