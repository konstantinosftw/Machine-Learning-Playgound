#  -*- coding: utf-8 -*-

from PRIVATE import *
import tweepy
from textblob import TextBlob
import pandas as pd

# authenticate app with twitter
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

# search twitter from API
public_tweets = api.search('\Led Zeppelin', count=100)

# create lists of tweets & sentiments
tweets = []
sentiment = []

for tweet in public_tweets:
    tweets.append(tweet.text)
    analysis = TextBlob(tweet.text)
    sentiment.append(analysis.sentiment.polarity)

# print(tweets)
# print(sentiment)

# prepare data
data = {'Tweets': tweets, 'Sentiment': sentiment}

df = pd.DataFrame(data, columns=['Tweets', 'Sentiment'])
df.to_excel('twitter play.xlsx')
