import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, recall_score

# Load the dataset from a CSV file
# Replace 'path_to_your_file.csv' with the actual file path
df = pd.read_csv('../Materias/NoT_NoBalance_Filtered_Total_Data2.csv')

# Using 'clean_text' for text and 'category' for labels
texts = df['clean_text']
labels = df['category']

# Feature extraction
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)
y = labels

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the SVM model
svm_model = SVC()
svm_model.fit(X_train, y_train)

# Making predictions
y_pred = svm_model.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

# Output the results
print(f'Accuracy: {accuracy:.2f}')
print(f'F1-score: {f1:.2f}')
print(f'Recall: {recall:.2f}')
