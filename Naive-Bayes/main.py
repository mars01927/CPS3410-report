import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from wordcloud import WordCloud
start_time = time.time()

data = pd.read_csv("../Materias/Twitter_Data.csv")
data['category'].mask(data['category'] == -1,'negative',  inplace=True)
data['category'].mask(data['category'] == 0,'neutral',  inplace=True)
data['category'].mask(data['category'] == 1,'positive',  inplace=True)

#This section is only used for process display, and the final running time does not calculate the process display time
print(data.head())

fig = plt.figure(figsize=(7, 5))
sns.countplot(x="category", data=data)
plt.title("Dataset labels distribuition")
plt.show()

fig = plt.figure(figsize=(7, 7))
colors = ("yellowgreen", "gold", "red")
wp = {'linewidth': 2, 'edgecolor': "black"}
tags = data['category'].value_counts()
explode = (0.1, 0.1, 0.1)
tags.plot(kind='pie', autopct='%1.1f%%', shadow=True, colors=colors,
          startangle=90, wedgeprops=wp, explode=explode, label='')
plt.title('Distribution of sentiments')
plt.show()


# positive
clean_data = data[data['category'] == 1].dropna(subset=['clean_text'])

text = " ".join(i for i in clean_data['clean_text'])

wordcloud = WordCloud(background_color="white").generate(text)

plt.figure(figsize=(15, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title('Word Cloud for Positive Reviews')
plt.show()

# negative
clean_data = data[data['category'] == -1].dropna(subset=['clean_text'])

text = " ".join(i for i in clean_data['clean_text'])

wordcloud = WordCloud(background_color="white").generate(text)

plt.figure(figsize=(15, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title('Word Cloud for Negative Reviews')
plt.show()

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report,  ConfusionMatrixDisplay

X = data['clean_text'].values.astype('U')
y = data['category'].values.astype('U')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify = y)

pipe = Pipeline([('tfidf_vectorizer', TfidfVectorizer(lowercase=True,
                                                      stop_words='english',
                                                      analyzer='word')),

                 ('naive_bayes', MultinomialNB())])

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

pipe.fit(list(X_train), list(y_train))
y_pred = pipe.predict(X_test)
print(confusion_matrix(y_pred, y_test))
print(accuracy_score(y_pred, y_test))
pipe['naive_bayes']

end_time = time.time()
total_time = end_time - start_time
print(f"Total runtime of the script: {total_time} seconds")