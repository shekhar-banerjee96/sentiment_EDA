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
geolocator = Nominatim(user_agent = "sentiment")
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
        
      
    popular_tweets = tw.Cursor(api.search_tweets,q=input_sms,lang="en",tweet_mode="extended").items(20)
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
        fig , ax = plt.subplots()
        sns.countplot(data = df ,y = 'Countries',orient="v" ,order = df['Countries'].value_counts().iloc[:20].index)    
        plt.title('Top Contributing Countries')
        plt.ylabel('Countries', fontsize=12)
        plt.xlabel('Frequency of Comments', fontsize=12)
        st.pyplot(fig)
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
        
        fig = plt.figure(figsize = (10,8))            
        sns.countplot(data = df ,y = 'Countries',orient="v",hue = 'reaction',order = df['Countries'].value_counts().iloc[:20].index)
        plt.title('Positive / Negative Reviews Distribution')
        plt.ylabel('Countries', fontsize=12)
        plt.xlabel('Frequency of Comments', fontsize=12)
        st.pyplot(fig)
        plt.clf()
        


 


    col1, col2,col3 = st.columns([1,1,1])


    with col1:
        body = ''.join(df[df['reaction'] == 'Pos']['text'])
        wc = WordCloud(width = 500 , height = 500).generate(body)
        plt.imshow(wc,interpolation='bilinear')
        plt.axis("off")
        st.header("Positive Comment Wordcloud")
        st.pyplot()
        
        plt.clf()

    with col2:
        ody = ''.join(df[df['reaction'] == 'Neg']['text'])
        wc = WordCloud(width = 500 , height = 500).generate(body)
        plt.imshow(wc,interpolation='bilinear')
        plt.axis("off")
        st.header("Negative Comment Wordcloud")
        st.pyplot()
        
        plt.clf()

    with col3:
        palette_color = sns.color_palette('bright') 
        fig, ax = plt.subplots()
        print(Counter(df.reaction))
        ax.pie(Counter(df.reaction).values(), labels=['Negative','Positive'], colors=palette_color, autopct='%.0f%%',radius=0.96)
        st.header('Distribution of Positive / Negative')
        st.pyplot(fig)
        plt.clf()
    



