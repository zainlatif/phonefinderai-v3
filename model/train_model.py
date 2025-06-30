import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Load data
df = pd.read_csv('../data/training_data.csv')

# Fill NaN with empty string
df['price'] = df['price'].fillna('')
df['brand'] = df['brand'].fillna('')

# Combine features for vectorization
df['combined'] = df['query'] + ' ' + df['price'] + ' ' + df['brand']

# Split data
X_train, X_test, y_train, y_test = train_test_split(df['combined'], df['label'], test_size=0.2, random_state=42)

# Vectorize text
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
clf = LogisticRegression()
clf.fit(X_train_vec, y_train)

# Evaluate
y_pred = clf.predict(X_test_vec)
print(classification_report(y_test, y_pred))

# Save model and vectorizer
joblib.dump(clf, '../model/classifier.pkl')
joblib.dump(vectorizer, '../model/vectorizer.pkl')
