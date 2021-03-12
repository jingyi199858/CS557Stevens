import pandas as pd
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.stem import LancasterStemmer, WordNetLemmatizer, PorterStemmer
from wordcloud import WordCloud, STOPWORDS
from textblob import TextBlob

amz_reviews = pd.read_csv("1429_1.csv")

columns = ['id', 'name', 'keys', 'manufacturer', 'reviews.dateAdded', 'reviews.date', 'reviews.didPurchase',
           'reviews.userCity', 'reviews.userProvince', 'reviews.dateSeen', 'reviews.doRecommend', 'asins',
           'reviews.id', 'reviews.numHelpful', 'reviews.sourceURLs', 'reviews.title']

df = pd.DataFrame(amz_reviews.drop(columns, axis=1, inplace=False))

# print(amz_reviews.shape)
# print(amz_reviews.columns)

# print(df.shape)

#df['reviews.rating'].value_counts().plot(kind='bar')
df['reviews.text'] = df['reviews.text'].astype(str)
print(df['reviews.text'][0])
def senti(x):
    return TextBlob(x).sentiment.polarity

df['senti_score'] = df['reviews.text'].apply(senti)

print(df.senti_score)
f = open("result.txt", "w")
f.write(df.senti_score.to_string())
f.close()