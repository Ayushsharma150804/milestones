import pandas as pd


df = pd.read_csv('Salary_Data.csv')


print("Dataset Preview:")
print(df.head())


def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers


outliers_experience = detect_outliers_iqr(df, 'YearsExperience')
outliers_salary = detect_outliers_iqr(df, 'Salary')


print("\nOutliers in YearsExperience:")
print(outliers_experience)

print("\nOutliers in Salary:")
print(outliers_salary)
