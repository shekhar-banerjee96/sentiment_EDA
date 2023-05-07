#import Libraries
import streamlit as st
st.set_page_config(layout="wide")
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
from PIL import Image
import seaborn as sns
import tweepy as tw
#import hydralit_components as hc
import time


#Do necessary downloads and initliate classes 
nltk.download('punkt')
nltk.download('stopwords')
plt.rcParams.update({'axes.facecolor':'black'})
plt.style.use('seaborn-v0_8-dark-palette')
stop_words = set(stopwords.words('english'))
geolocator = Nominatim(user_agent = "geoapiExercises")
ps = PorterStemmer()

#Set Credentials for Tweepy API
#import configparser

#config = configparser.ConfigParser()
#config.read_file(open(r'cred.cfg'))
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
     
    try:
        location = geolocator.geocode(x,language='en',timeout=None)
        return str(location.raw['display_name'].split(',')[-1]).strip()
    except :
        return None
    
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
if st.button('Print Report'):
    # a dedicated single loader 
        
      
    popular_tweets = tw.Cursor(api.search_tweets,q=input_sms,lang="en",tweet_mode="extended").items(200)
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
    col1, col2 = st.columns([1,1])

    with col1:
        # Count plot to show the distribution of comments between the various countries whose people tweeted
        with sns.axes_style("darkgrid"):
            plt.figure(figsize = (10,8), facecolor = None)
            sns.countplot(data = df ,y = 'Countries',orient="v",order = df['Countries'].value_counts().iloc[:20].index)    
        time = datetime.now().strftime("%H%M%S")    
        fname = input_sms.replace('#','') + time + '.jpg'
        plt.savefig(fname,bbox_inches = 'tight')
        st.header("Top Contributing Countries")
        image = Image.open(fname)
        st.image(fname)
        plt.clf()

        # Feature to use emotion classifier to segregate emotions and disply the chart showing the split

        # loading trained classifier that can differentaite between postive and negative tweets
        # Its has been trained ober 1.6 million data records conataining of real time tweets.
        # dataset picked from kaggle

    df['cleaned_text'] = df['text'].apply(lambda x :  basic_clean(x,'simple'))
    model = pickle.load(open('model/sgdc.pkl', 'rb'))
    pipe = pickle.load(open('model/pipe.pkl', 'rb'))

    # prediction of tweets and providing them string labels for better visualizations
    corpus_vector = pipe.transform(df['cleaned_text']).toarray()
    df['reaction'] = model.predict(corpus_vector)
    df['reaction'] = df['reaction'].apply(lambda x : 'Pos' if x == 1 else 'Neg')

        
        
        
    with col2:   
        with sns.axes_style("darkgrid"):
            plt.figure(figsize = (10,8), facecolor = None)            
            sns.countplot(data = df ,y = 'Countries',orient="v",hue = 'reaction',order = df['Countries'].value_counts().iloc[:20].index)
            
        

        time = datetime.now().strftime("%H%M%S")  
        fname = input_sms.replace('#','') + time + '_pos_neg_dist.jpg'  
        plt.savefig(fname,bbox_inches = 'tight')
        st.header("Positive / Negative Reviews Distribution")
        image = Image.open(fname)
        st.image(fname, caption='Positive / Negative Reviews Distribution')
        plt.clf()

 


    col1, col2,col3 = st.columns([1,1,1])


    with col1:
        body = ''.join(df[df['reaction'] == 'Pos']['text'])
        wc = WordCloud(width = 500 , height = 500).generate(body)
        plt.figure(figsize = (12, 8), facecolor = None)
        plt.imshow(wc)
        st.header("Negative Comment Wordcloud")
        time = datetime.now().strftime("%H%M%S")  
        fname = input_sms.replace('#','') + time + '_neg_wordcloud.jpg'  
        plt.savefig(fname,bbox_inches = 'tight')
        image = Image.open(fname)
        st.image(fname)
        plt.clf()

    with col2:
        body = ''.join(df[df['reaction'] == 'Pos']['text'])
        wc = WordCloud(width = 500 , height = 500).generate(body)
        plt.figure(figsize = (12, 8), facecolor = None)
        plt.imshow(wc)
        st.header("Positive Comment Wordcloud")
        time = datetime.now().strftime("%H%M%S")  
        fname = input_sms.replace('#','') + time + '_pos_wordcloud.jpg'  
        plt.savefig(fname,bbox_inches = 'tight')
        image = Image.open(fname)
        st.image(fname)
        plt.clf()
        image.close()

    with col3:
        palette_color = sns.color_palette('bright')
        plt.figure(figsize = (12,8), facecolor = None)
        plt.pie(Counter(df.reaction).values(), labels=['Negative','Positive'], colors=palette_color, autopct='%.0f%%',radius=0.96)
        plt.title('Distribution of Positive / Negative')
        
        time = datetime.now().strftime("%H%M%S")  
        fname = input_sms.replace('#','') + time + '_total_pos_neg_dist.jpg'  
        plt.savefig(fname,bbox_inches = 'tight')
        st.header(" Positive / Negative Distribution For Total")
        
        image = Image.open(fname)
        st.image(fname)
        plt.clf()
        image.close()


import os

files = os.listdir()

for i in files:
    if (i.split(".")[-1] == 'jpg') :
        os.remove(i)
    



