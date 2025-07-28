#cars.csV


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


df = pd.read_csv("cars.csv")


df.drop(['car name'], axis=1, inplace=True)


df.replace("?", np.nan, inplace=True)
df = df.dropna()


df['horsepower'] = df['horsepower'].astype(float)


X = df.drop('mpg', axis=1)
y = df['mpg']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


print(" R^2 Score:", r2_score(y_test, y_pred))
print(" MAE:", mean_absolute_error(y_test, y_pred))
print(" MSE:", mean_squared_error(y_test, y_pred))
print(" RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))


plt.scatter(y_test, y_pred, color='green')
plt.xlabel("Actual MPG")
plt.ylabel("Predicted MPG")
plt.title("Actual vs Predicted MPG")
plt.grid(True)
plt.show()

#50_Startups.csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


df = pd.read_csv("50_Startups.csv")


print("First 5 rows:\n", df.head())
print("\nColumn names:", df.columns)


df = pd.get_dummies(df, drop_first=True)

X = df.drop('Profit', axis=1)
y = df['Profit']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


print("\n R² Score:", r2_score(y_test, y_pred))
print(" Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print(" Mean Squared Error:", mean_squared_error(y_test, y_pred))
print(" Root Mean Squared Error:", np.sqrt(mean_squared_error(y_test, y_pred)))


plt.scatter(y_test, y_pred, color='blue')
plt.xlabel("Actual Profit")
plt.ylabel("Predicted Profit")
plt.title("Actual vs Predicted Profit")
plt.grid(True)
plt.show()

#salary_data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


df = pd.read_csv("Salary_Data.csv")


print("First 5 rows:\n", df.head())
print("\nDataset Info:")
print(df.info())


X = df[['YearsExperience']]
y = df['Salary']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


print("\n R² Score:", r2_score(y_test, y_pred))
print(" MAE:", mean_absolute_error(y_test, y_pred))
print(" MSE:", mean_squared_error(y_test, y_pred))
print(" RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))


plt.scatter(X, y, color='blue', label='Actual')
plt.plot(X, model.predict(X), color='red', label='Predicted Line')
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Salary vs Experience")
plt.legend()
plt.grid(True)
plt.show()



#Logistic Regression and Evaluation Metrics

#Logistic Regression and Evaluation Metrics


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


df = pd.read_csv("cancer.csv")


print(" Columns in dataset:")
print(df.columns)


target_column = 'target'


if 'id' in df.columns:
    df.drop('id', axis=1, inplace=True)
if 'Unnamed: 32' in df.columns:
    df.drop('Unnamed: 32', axis=1, inplace=True)


if df[target_column].dtype == 'object':
    df[target_column] = df[target_column].map({'M': 1, 'B': 0})  # Example for diagnosis


X = df.drop(target_column, axis=1)
y = df[target_column]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n Classification Report:")
print(classification_report(y_test, y_pred))

print(" Accuracy Score:", accuracy_score(y_test, y_pred))
print("\n Confusion Matrix:")
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


#Social_Network_Ads


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


df = pd.read_csv("Social_Network_Ads.csv")


print("📄 Dataset Head:")
print(df.head())
print("\n📊 Columns:")
print(df.columns)
print("\n🔍 Dataset Info:")
print(df.info())



if 'User ID' in df.columns:
    df.drop('User ID', axis=1, inplace=True)


if df['Gender'].dtype == 'object':
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})


X = df.drop('Purchased', axis=1)  # Features
y = df['Purchased']              # Target


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)


model = LogisticRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


print("\n Accuracy:", accuracy_score(y_test, y_pred))
print("\n Classification Report:\n", classification_report(y_test, y_pred))


cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

