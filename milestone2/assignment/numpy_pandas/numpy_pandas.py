

import numpy as np

array_2d = np.array([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9]])

print("Array:\n", array_2d)
print("Number of dimensions (ndim):", array_2d.ndim)
print("Shape of the array:", array_2d.shape)
print("First row:", array_2d[0])
print("Second column:", array_2d[:, 1])
print("Element at row 2, col 3:", array_2d[1, 2])
print("Top-left 2x2 sub-array:\n", array_2d[0:2, 0:2])




#Create one dimensional array and perform ndim, shape, reshape operation on it.
import numpy as np

array_1d = np.array([10, 20, 30, 40, 50, 60])

print("Original 1D Array:", array_1d)

print("ndim:", array_1d.ndim)

print("shape:", array_1d.shape)

reshaped_array = array_1d.reshape(2, 3)
print("Reshaped to 2D (2x3):\n", reshaped_array)

reshaped_array2 = array_1d.reshape(3, 2)
print("Reshaped to 2D (3x2):\n", reshaped_array2)

#pandas

#Download the data set and rename to cars.csv
import pandas as pd

cars = pd.read_csv(r"C:\Users\Prince\PycharmProjects\PythonProject/cars.csv")

print("First 10 rows:")
print(cars.head(10))

print("\nEntire DataFrame (first 50 rows shown for brevity):")
print(cars.head(50))

print("\nLast 5 rows:")
print(cars.tail())

print("\nMeta Information:")
print(cars.info())


# Download 50 startups dataset


df = pd.read_csv(r"C:\Users\Prince\PycharmProjects\PythonProject\50_Startups.csv")

print("First 5 rows:")
print(df.head())

print("\n Statistical Summary:")
print(df.describe())

# Drop non-numeric column for correlation
df_numeric = df.drop("State", axis=1)

print("\n Correlation Coefficients (numeric columns only):")
print(df_numeric.corr())









