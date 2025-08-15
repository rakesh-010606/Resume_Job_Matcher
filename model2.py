import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Loading dataset
df = pd.read_csv('job_applicant_dataset.csv')
df=df.sample(frac=1).reset_index(drop=True)  # shuffling the data to avoid bias

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Text preprocessing
stop_words=set(stopwords.words('english'))
lemmatizer=WordNetLemmatizer()

def preprocess(text):
    text=text.lower()
    text=text.translate(str.maketrans('','',string.punctuation))
    tokens=nltk.word_tokenize(text)
    tokens=[lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)

# Preprocess text columns
df['resume']=df['Resume'].apply(preprocess)
df['jobdesc']=df['Job Description'].apply(preprocess)


# TF-IDF vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf=TfidfVectorizer()
tfidf.fit(pd.concat([df['resume'], df['jobdesc']]))

resume_tfidf=tfidf.transform(df['resume'])
jobdesc_tfidf=tfidf.transform(df['jobdesc'])

# Label encode categorical columns
from sklearn.preprocessing import LabelEncoder

le_gender=LabelEncoder()
le_race=LabelEncoder()
df['Gender_enc']=le_gender.fit_transform(df['Gender'])
df['Race_enc']=le_race.fit_transform(df['Race'])
df['Ethnicity_enc']=le_race.fit_transform(df['Ethnicity'])

extra_features=df[['Age', 'Gender_enc', 'Race_enc', 'Ethnicity_enc']].values #numpy format

# Convert extra features to sparse matrix
from scipy import sparse

extra_sparse=sparse.csr_matrix(extra_features) # sparse matrix format

X=sparse.hstack([resume_tfidf, jobdesc_tfidf, extra_sparse])# Combining all features horizontally
y=df['Best Match']

# Train-test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2)

# Train Random Forest
from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier()
rf.fit(X_train, y_train)

# Prediction and evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

y_pred=rf.predict(X_test) # predicting using randomforest classifier

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))


