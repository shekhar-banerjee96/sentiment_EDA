import streamlit as st
st.set_page_config(layout="wide")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

fig , ax = plt.subplots()
df = sns.load_dataset("penguins")
sns.countplot(data = df,x = 'island',ax=ax,orient = "v",order = df['island'].value_counts().index)
st.pyplot(fig)

#import Libraries
import streamlit as st

import pandas as pd
import nltk ,emoji , re ,string , pickle
from nltk.corpus import stopwords
from datetime import datetime
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from geopy.geocoders import Nominatim
from nltk.stem import PorterStemmer
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
import tweepy as tw
#import hydralit_components as hc
import time

st.set_option('deprecation.showPyplotGlobalUse', False)
#Do necessary downloads and initliate classes 
nltk.download('punkt')
nltk.download('stopwords')
plt.rcParams.update({'axes.facecolor':'black'})
sns.set_style("darkgrid")

stop_words = set(stopwords.words('english'))
geolocator = Nominatim(user_agent = "geoapiExercises")
ps = PorterStemmer()

#Set Credentials for Tweepy API

consumer_key = st.secrets['cred']['consumer_key']
consumer_secret = st.secrets['cred']['consumer_secret']
access_token = st.secrets['cred']['access_token']
access_secret = st.secrets['cred']['access_secret']

# Authenticate to Twitter
auth = tw.OAuthHandler(consumer_key,consumer_secret)
auth.set_access_token(access_token,access_secret)

api = tw.API(auth,wait_on_rate_limit=True)

try:
    api.verify_credentials()
    print("Authentication OK")
except:
    print("Error during authentication")

df = pd.DataFrame(columns=['created_at', 'Location', 'text'])

st.title("Twitter Sentiment EDA")

#Enter you search string
input_sms = st.text_area("Enter the hastag example #India , #Tesla")




def countries(x):
    #This function will use the Geolocator API and find the Country of the user. 
    # we split the reslt and provide the last part which contains tge country name.
  location = geolocator.geocode(x,language='en',timeout=None)
  return str(location.raw['display_name'].split(',')[-1]).strip()


st.header(countries('Delhi'))
    
def basic_clean(x,type) :

  ignore  = ['rt','https','u']  
  words  = emoji.demojize(x.lower())
   
  
  words = re.sub('[^a-zA-Z0-9\n\.] | @\\w+', ' ', "".join(words))
  
  words = word_tokenize(words)
  final = []
  if (type == 'simple'):
    for word in words :
      if (word not in stop_words) and (word not in string.punctuation) and (word not in ignore)  and word.isalnum():
        final.append(ps.stem(word))
  elif (type == 'wc'):
    for word in words :
      if (word not in stop_words) and (word not in string.punctuation) and (word not in ignore)  and word.isalnum():
        final.append(word)

  words =  ' '.join(final[:])
  
  return words


#This button initiate the Data Import using the input tring from above 

        
      
popular_tweets = tw.Cursor(api.search_tweets,q='#china',lang="en",tweet_mode="extended").items(10)
for tweet in popular_tweets:
    #pull data fields as per requirements , for now pulling timestamp which will act like id , user name , location , tweet text
    # we will geolocator api to find the country name from the location data.

    delta = pd.DataFrame({
        'text': basic_clean(tweet._json['full_text'],'wc')
        ,'Countries': countries(tweet._json['user']['location'])
        ,'cleaned_text': basic_clean(tweet._json['full_text'],'simple')
                            }, index= [tweet.id])
    df = pd.concat([delta , df ])

print('Data Imported')
st.dataframe(df)
