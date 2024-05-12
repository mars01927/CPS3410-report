import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score, f1_score, recall_score

# Load the dataset from a CSV file
# Replace 'path_to_your_file.csv' with the actual file path
try:
    print("Loading data with UTF-8 encoding...")
    df = pd.read_csv('../Materias/NoT_NoBalance_Filtered_Total_Data2.csv', encoding='utf-8')
except UnicodeDecodeError:
    print("UTF-8 encoding failed, trying ISO-8859-1 encoding...")
    df = pd.read_csv('../Materias/NoT_NoBalance_Filtered_Total_Data2.csv', encoding='ISO-8859-1', errors='replace')

print("Data loaded successfully!")

# Using 'clean_text' for text and 'category' for labels
texts = df['clean_text']
labels = df['category']

# Adjust labels: map -1 to 0 (negative) and 1 to 1 (positive)
print("Mapping labels...")
labels = labels.map({-1: 0, 1: 1})
print("Labels mapped successfully!")

# Initialize VADER sentiment analyzer
print("Initializing VADER Sentiment Analyzer...")
analyzer = SentimentIntensityAnalyzer()

# Function to classify sentiment
def vader_sentiment_analysis(text):
    score = analyzer.polarity_scores(text)
    # Classify as positive (1) if compound score > 0, otherwise negative (0)
    return 1 if score['compound'] > 0 else 0

print("Applying VADER sentiment analysis to texts...")
# Apply VADER sentiment analysis to the texts
predictions = texts.apply(vader_sentiment_analysis)
print("Sentiment analysis completed!")

# Evaluating the model
print("Evaluating the model...")
accuracy = accuracy_score(labels, predictions)
f1 = f1_score(labels, predictions)
recall = recall_score(labels, predictions)

# Output the results
print(f'Accuracy: {accuracy:.2f}')
print(f'F1-score: {f1:.2f}')
print(f'Recall: {recall:.2f}')
