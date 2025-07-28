import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

data = {
    'text': [
        "Excellent food and great service!",
        "Very bad experience, not recommended.",
        "Okay place, not too bad.",
        "Loved the ambiance and staff was friendly.",
        "Horrible taste, waste of money.",
        "Mediocre service and average food.",
        "Best restaurant in town!",
        "Would not go again, poor quality.",
        "Food was decent, nothing special.",
        "Absolutely fantastic! Highly recommend it."
    ],
    'stars': [5, 1, 3, 5, 1, 3, 5, 2, 3, 5]
}

df = pd.DataFrame(data)

X = df['text']
y = df['stars']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
