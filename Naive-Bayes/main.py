import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from wordcloud import WordCloud
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

start_time = time.time()

# Load data
data = pd.read_csv("../Materias/Filtered_Total_Data2.csv")

# Convert category numbers to string labels
data['category'].mask(data['category'] == -1, 'negative', inplace=True)
data['category'].mask(data['category'] == 0, 'neutral', inplace=True)
data['category'].mask(data['category'] == 1, 'positive', inplace=True)

# Function to generate word clouds
def generate_wordcloud(category_label):
    clean_data = data[data['category'] == category_label].dropna(subset=['clean_text'])
    text = " ".join(i for i in clean_data['clean_text'])
    if text.strip():  # Check if text is not empty
        wordcloud = WordCloud(background_color="white").generate(text)
        plt.figure(figsize=(15, 10))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title(f'Word Cloud for {category_label.capitalize()} Reviews')
        plt.show()
    else:
        print(f"No text available for {category_label} word cloud.")

# Generate word clouds
generate_wordcloud('positive')
generate_wordcloud('negative')

# Display data distribution
fig = plt.figure(figsize=(7, 5))
sns.countplot(x="category", data=data)
plt.title("Dataset labels distribution")
plt.show()

# Prepare data for modeling
X = data['clean_text'].values.astype('U')
y = data['category'].values.astype('U')

# Handle class imbalance using SMOTE
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X.reshape(-1, 1), y)
X_resampled = X_resampled.flatten()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, stratify=y_resampled)

# Build pipeline with hyperparameter tuning
pipeline = Pipeline([
    ('tfidf_vectorizer', TfidfVectorizer(lowercase=True, stop_words='english', analyzer='word')),
    ('naive_bayes', MultinomialNB())
])

# Define hyperparameters to tune
param_grid = {
    'tfidf_vectorizer__max_df': [0.75, 0.85, 1.0],
    'tfidf_vectorizer__ngram_range': [(1, 1), (1, 2)],
    'naive_bayes__alpha': [0.01, 0.1, 1]
}

# Perform Grid Search with Stratified K-Fold
cv = StratifiedKFold(n_splits=5)
grid_search = GridSearchCV(pipeline, param_grid, cv=cv, n_jobs=-1, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best parameters
print("Best parameters found:", grid_search.best_params_)

# Evaluate best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Evaluate model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test)
plt.show()

accuracy = accuracy_score(y_test, y_pred) * 100
f1 = f1_score(y_test, y_pred, average='macro')
print(f"Accuracy: {accuracy:.2f}%")
print(f"F1 Score: {f1:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

end_time = time.time()
total_time = end_time - start_time
print(f"Total runtime of the script: {total_time:.2f} seconds")
