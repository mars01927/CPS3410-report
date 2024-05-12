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
from sklearn.svm import SVC
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

start_time = time.time()

# Load data
data = pd.read_csv("../Materias/NoT_NoBalance_Filtered_Total_Data2.csv")

# Convert category numbers to string labels
data['category'].replace({-1: 'negative', 0: 'neutral', 1: 'positive'}, inplace=True)

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

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# Build and train model pipeline with SMOTE
pipeline = ImbPipeline([
    ('tfidf_vectorizer', TfidfVectorizer(lowercase=True, stop_words='english', analyzer='word')),
    ('smote', SMOTE()),
    ('svm', SVC())
])

# Define hyperparameters to tune
param_grid = {
    'tfidf_vectorizer__max_df': [0.75, 0.85, 1.0],
    'tfidf_vectorizer__ngram_range': [(1, 1), (1, 2)],
    'svm__C': [0.1, 1, 10],
    'svm__gamma': ['scale', 'auto'],
    'svm__kernel': ['linear', 'rbf']
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
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)
ConfusionMatrixDisplay(conf_matrix, display_labels=best_model.named_steps['svm'].classes_).plot()
plt.show()

accuracy = accuracy_score(y_test, y_pred) * 100
f1 = f1_score(y_test, y_pred, average='macro')
print(f"Accuracy: {accuracy:.2f}%")
print(f"F1 Score: {f1:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=np.unique(y)))

end_time = time.time()
total_time = end_time - start_time
print(f"Total runtime of the script: {total_time:.2f} seconds")
