#!/usr/bin/env python
# coding: utf-8

# In[1]:


#yelp_checkin = pd.read_json(r'C:\Users\Owais\Documents\Data Science Course\CIND820\yelp_dataset\yelp_academic_dataset_checkin.json', lines = True)
#yelp_tip = pd.read_json(r'C:\Users\Owais\Documents\Data Science Course\CIND820\yelp_dataset\yelp_academic_dataset_tip.json', lines = True)
#yelp_user = pd.read_json(r'C:\Users\Owais\Documents\Data Science Course\CIND820\yelp_dataset\yelp_academic_dataset_user.json', lines = True)


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random


# In[3]:


yelp_business = pd.read_json(r'C:\Users\Owais\Documents\Data Science Course\CIND820\yelp_dataset\yelp_academic_dataset_business.json', lines = True)


# In[4]:


business = yelp_business

business = business[['business_id', 'name', 'is_open', 'stars', 'review_count', 'categories', 'attributes', 'state', 'city', 'latitude', 'longitude']]
business.head()


# In[5]:


yelp_review = pd.read_json(r'C:\Users\Owais\Documents\Data Science Course\CIND820\yelp_dataset\yelp_academic_dataset_review.json', lines = True)


# In[6]:


review = yelp_review


# In[7]:


review.describe().apply(lambda s: s.apply('{0:.2f}'.format))


# In[8]:


yelp_review['text length'] = yelp_review['text'].apply(len)
yelp_review.head()


# # Using Open Businesses in City

# In[5]:


open_business = business[business['is_open']==1].reset_index(drop = True)

open_business.groupby(['city']).agg({'review_count':'sum', 
                         'business_id':'size'}).sort_values(by = 'review_count', ascending = False).head(10)


# In[10]:


business_in_Cambridge = open_business[open_business["city"] == "Cambridge"].reset_index(drop = True)


# In[12]:


business_in_Cambridge.rename({"stars": "business_rating"}, axis="columns", inplace = True)
business_in_Cambridge["business_rating"].value_counts().sort_index()


# In[13]:


x = business_in_Cambridge['business_rating'].value_counts().sort_index().index
y = business_in_Cambridge['business_rating'].value_counts().sort_index().values
fig, ax = plt.subplots()
plt.bar(x, y)
plt.xlabel('Business Ratings')
plt.ylabel('Count of Businesses')
plt.show()


# In[14]:


cols_to_keep = ['business_id', 'name', 'city', 'state','latitude', 'longitude', 'business_rating']
Cambridge_business_reviews = pd.merge(business_in_Cambridge[cols_to_keep], review, left_on = ['business_id']                                     , right_on = ['business_id'], how = 'left')


# In[15]:


Cambridge_business_reviews.head()


# In[16]:


len(Cambridge_business_reviews['text'])


# In[17]:


#Check Null values
Cambridge_business_reviews.isnull().sum()


# # Lowercasing and Removing Punctuations & Stopwords

# In[18]:


Cambridge_business_reviews['lowercase'] = Cambridge_business_reviews['text'].apply(lambda x: " ".join(word.lower() for word in x.split()))
Cambridge_business_reviews['punctuation'] = Cambridge_business_reviews['lowercase'].str.replace('[^\w\s]','')


# In[19]:


import spacy
from spacy.lang.en.stop_words import STOP_WORDS

#from nltk.corpus import stopwords


# In[20]:


stopwords = STOP_WORDS


# In[21]:


Cambridge_business_reviews['stopwords'] = Cambridge_business_reviews['punctuation'].apply(lambda x: " ".join(word for word in x.split() if word not in stopwords))


# In[22]:


pd.Series(" ".join(Cambridge_business_reviews['stopwords']).split()).value_counts()[:60]


# In[23]:


other_stopwords = ['im', 'ive', 'get', 'told', 'would', 'us', 'go', 'sure', 'like', 'came', 'didnt', 'told', 'way', 'place' ]


# In[24]:


Cambridge_business_reviews['cleaned_reviews'] = Cambridge_business_reviews['stopwords'].apply(lambda x: " ".join(word for word in x.split() if word not in other_stopwords))


# In[25]:


pd.Series(" ".join(Cambridge_business_reviews['cleaned_reviews']).split()).value_counts()[:20]


# In[26]:


Cambridge_business_reviews['cleaned_reviews_text_length'] = Cambridge_business_reviews['cleaned_reviews'].apply(lambda x: len(x.split()))
Cambridge_business_reviews['clean_rate'] = Cambridge_business_reviews['cleaned_reviews_text_length'] / Cambridge_business_reviews['text length']


# In[27]:


Cambridge_business_reviews.head()


# # Lemmatization

# In[28]:


import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
from textblob import Word


# In[29]:


Cambridge_business_reviews['lemmatized'] = Cambridge_business_reviews['cleaned_reviews'].apply(lambda x: " ".join(Word(word).lemmatize() for word in x.split()))


# In[30]:


Cambridge_business_reviews.head()


# # Sentiment Analysis

# In[31]:


from textblob import TextBlob


# In[32]:


Cambridge_business_reviews['Polarity'] = Cambridge_business_reviews['lemmatized'].apply(lambda x:TextBlob(x).sentiment[0])
Cambridge_business_reviews['Subjectivity'] = Cambridge_business_reviews['lemmatized'].apply(lambda x:TextBlob(x).sentiment[1])


# In[33]:


Cambridge_business_reviews.drop(['lowercase', 'punctuation', 'stopwords'], axis=1, inplace=True)


# In[34]:


Cambridge_business_reviews.sort_values(by='name')


# In[35]:


Cambridge_business_reviews['Sentiment_rating'] = np.where(Cambridge_business_reviews['Polarity'] >= 0, 1, 0)
Cambridge_business_reviews.head()


# In[36]:


Cambridge_business_reviews.describe().apply(lambda s: s.apply('{0:.2f}'.format))


# In[37]:


Cambridge_business_reviews[['Sentiment_rating']].describe().apply(lambda s: s.apply('{0:.2f}'.format))


# # Word Cloud

# In[38]:


#pip install wordcloud


# In[39]:


from wordcloud import WordCloud


# In[40]:


# Dividing the reviews to positive and negative
pos = Cambridge_business_reviews[Cambridge_business_reviews['Sentiment_rating']==1]['text']
neg = Cambridge_business_reviews[Cambridge_business_reviews['Sentiment_rating']==0]['text']


# In[60]:


pos_text = " ".join(i for i in pos)

pos_wc = WordCloud(background_color='white', max_words=100)
pos_wc.generate(pos_text)
plt.figure(figsize=(10,7))
plt.imshow(pos_wc, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[61]:


neg_text = " ".join(i for i in neg)

neg_wc = WordCloud(background_color='white', max_words=100)
neg_wc.generate(neg_text)
plt.figure(figsize=(10,7))
plt.imshow(neg_wc, interpolation='bilinear')
plt.axis("off")
plt.show()


# # Machine Learning

# In[41]:


#Splitting the data to train and test
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[42]:


X = Cambridge_business_reviews["text"]
y = Cambridge_business_reviews["Sentiment_rating"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)


# In[43]:


y.value_counts()


# In[44]:


X_train.shape, X_test.shape, y_train.shape, y_test.shape


# # Vectorization

# In[45]:


from sklearn.feature_extraction.text import TfidfVectorizer
vect_1 = TfidfVectorizer(min_df=100, ngram_range=(1,3)).fit(X_train)
X_train1 = vect_1.transform(X_train)
X_test1 = vect_1.transform(X_test)


# In[46]:


X_train1.shape, X_test1.shape


# # Balancing

# In[47]:


from imblearn.over_sampling import SMOTE


# In[48]:


#SMOTE the training data
sm = SMOTE(random_state=1)
X_bal, y_bal = sm.fit_resample(X_train1, y_train)


# In[49]:


y_bal.value_counts()


# # Logistic Regression

# In[50]:


# fitting a logistic regression model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings("ignore")

# Fitting Logistic regression to the training set
logreg = LogisticRegression(solver='lbfgs',multi_class='auto',random_state=1)
logreg.fit(X_bal, y_bal)

# Predicting the test set results
y_pred_logreg = logreg.predict(X_test1)

# Training score
print(f"Score on training set: {logreg.score(X_train1,y_train)}")
print(f"Score on test set: {logreg.score(X_test1,y_test)}")


# In[51]:


from sklearn.metrics import classification_report
print('The Confusion Matrix')
con_mat_lr = confusion_matrix(y_test, y_pred_logreg)
df_cm_lr = pd.DataFrame(con_mat_lr, columns = ['Predicted 0','Predicted 1'], index = ['True 0','True 1'])
display(df_cm_lr)
print('The Classification report')
report = classification_report(y_test, y_pred_logreg, output_dict=True)
df_report = pd.DataFrame(report).transpose()
df_report


# # Naive Bayes

# In[52]:


from sklearn.naive_bayes import MultinomialNB

nb_classifier = MultinomialNB()
nb_classifier.fit(X_bal, y_bal)
nb_classifier.predict(X_test1)

# Predicting the test set results
y_pred_nb = nb_classifier.predict(X_test1)

# Training score
print(f"Score on training set: {nb_classifier.score(X_train1,y_train)}")
print(f"Score on test set: {nb_classifier.score(X_test1,y_test)}")


# In[53]:


from sklearn.metrics import classification_report
print('The Confusion Matrix')
con_mat_nb = confusion_matrix(y_test, y_pred_nb)
df_cm_nb = pd.DataFrame(con_mat_nb, columns = ['Predicted 0','Predicted 1'], index = ['True 0','True 1'])
display(df_cm_nb)
print('The Classification report')
report = classification_report(y_test, y_pred_nb, output_dict=True)
df_report = pd.DataFrame(report).transpose()
df_report


# # Random Forest

# In[62]:


from sklearn.ensemble import RandomForestClassifier

my_random_forest = RandomForestClassifier(random_state=42)
my_random_forest.fit(X_bal, y_bal)

# Training score
print(f"Score on training set: {my_random_forest.score(X_train1,y_train)}")
print(f"Score on test set: {my_random_forest.score(X_test1, y_test)}")


# In[63]:


# Predicting the test set results
y_pred_rand = my_random_forest.predict(X_test1)

# Creating confusion matrix/ dataFrame
print('The Confusion Matrix')
con_mat_RF = confusion_matrix(y_test, y_pred_rand)
df_cm_RF = pd.DataFrame(con_mat_RF, columns = ['Predicted 0','Predicted 1'], index = ['True 0','True 1'])
df_cm_RF


# In[64]:


print('The Classification report')
report = classification_report(y_test, y_pred_rand, output_dict=True)
df_report = pd.DataFrame(report).transpose()
df_report

