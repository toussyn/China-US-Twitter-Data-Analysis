#import python packages
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from wordcloud import WordCloud
import plotly.express as px 
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pickle
from  pages.plots import *

st.set_page_config(page_title="Dashboard", layout="wide")
loaded_df = None
def loadData():
    query = "select * from TweetInformation"
    # df = db_execute_fetch(query, dbName="tweets", rdf=True)
    df = pd.read_csv("./st_dashboard/processed_global_data_tweets.csv") #For deployed version
    loaded_df = df
    return df

def selectHashTag():
    df = loadData() if loaded_df is None else loaded_df
    hashTags = st.multiselect("choose combaniation of hashtags", list(df['hashtags'].unique()))
    if hashTags:
        df = df[np.isin(df, hashTags).any(axis=1)]
        st.write(df)

def selectLocAndAuth():
    df = loadData() if loaded_df is None else loaded_df
    location = st.multiselect("choose Location of tweets", list(df['place'].unique()))
    # lang = st.multiselect("choose Language of tweets", list(df['language'].unique()))
    lang = st.multiselect("choose Language of tweets", list(df['lang'].unique())) #For deployed version

    if location and not lang:
        df = df[np.isin(df, location).any(axis=1)]
        st.write(df)
    elif lang and not location:
        df = df[np.isin(df, lang).any(axis=1)]
        st.write(df)
    elif lang and location:
        location.extend(lang)
        df = df[np.isin(df, location).any(axis=1)]
        st.write(df)
    else:
        st.write(df)

def wordCloud():
    df = loadData() if loaded_df is None else loaded_df
    cleanText = ''
    for text in df['full_text']:
        tokens = str(text).lower().split()

        cleanText += " ".join(tokens) + " "

    wc = WordCloud(width=650, height=450, background_color='white', min_font_size=5).generate(cleanText)
    st.title("Tweet Text Word Cloud")
    st.image(wc.to_array())


def trainModelTest():
    st.markdown("<p style='padding:10px; background-color:#000320;color:#00ECB9;font-size:16px;border-radius:10px;'>Your CSV file should have 'full_text' as a column</p>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a CSV file")
    try:
        if uploaded_file is not None:
            print(uploaded_file.type)
            if uploaded_file.type == 'text/csv':
                df = pd.read_csv(uploaded_file)
                df['full_text'] = df['full_text'].apply(
                    lambda text: text.replace(',', ' ')
                )
                df['polarity'] = getPolarity(df['full_text'])
                processed_df = df[["full_text", "polarity"]]
                def applyConvert(val):
                    if val < 0:
                        return "negative"
                    elif val == 0:
                        return "neutral"
                    else:
                        return "positive"
                processed_df['score'] = processed_df['polarity'].apply(applyConvert)
                processed_df = processed_df[processed_df['score']!="neutral"]
                processed_df['scoremap'] = processed_df['score'].map(
                    lambda val: 1 if "positive" else 0
                )
                (X, y) = processed_df['full_text'], processed_df['scoremap']
                trigram_vectorizer = CountVectorizer(ngram_range=(1, 3))
                trigram_vectorizer.fit(X.values)
                X_trigram_vectorizer = trigram_vectorizer.transform(X.values)
                loaded_model = None
                loaded_X_test = None
                loaded_y_test = None
                with open('./st_dashboard/model.pkl', 'rb') as f:
                    loaded_model = pickle.load(f)
                with open("./st_dashboard/X_test.pkl", 'rb') as f:
                    loaded_X_test = pickle.load(f)


                with open("./st_dashboard/y_test.pkl", 'rb') as f:
                    loaded_y_test = pickle.load(f)
                if loaded_model is not None:
                    # print(X_trigram_vectorizer)
                    # print(loaded_model.predict(X_trigram_vectorizer))
                    test_score = loaded_model.score(loaded_X_test, loaded_y_test)
                    st.write(f'Test score: {round(test_score, 2)}')
                else:
                    st.write("No model found to test.")
        else:
            st.write("Only .csv file is allowed")
    except Exception as e:
        msg = str(e)
        print(msg)
        st.write(msg)


def getPolarity(text: str):
    polarity = []
    for t in text:
        each_sentiment = TextBlob(t).sentiment
        polarity.append(each_sentiment.polarity)
    return polarity


def findFullText(df: pd.DataFrame):
    print(len(df))
    text = [d['full_text'] for d in df]
    text = [d['full_text'].replace(',', ' ')
            for d in df.iteritems()]
    return text

def mainPage():
    st.title("Data Display")
    selectHashTag()
    st.markdown("<p style='padding:10px; background-color:#000000;color:#00ECB9;font-size:16px;border-radius:10px;'>Section Break</p>", unsafe_allow_html=True)
    selectLocAndAuth()
    wordCloud()
    # with st.expander("Show More Graphs"):
    #     locationPie()
    #     userMentionbarChart()
    #     sourcePie()
    #     stBarChart()
    #     sentimentPie() #Only For deployed version
    #     langPie()
    
    with st.expander("Test the trained model with a csv file"):
        trainModelTest()

def plots():
    # st.markdown("# Data Visualizations ❄️")
    st.sidebar.markdown("# Data Visualizations ❄️")
    st.title("Data Visualizations")

    locationPie()
    userMentionbarChart()
    sourcePie()
    stBarChart()
    sentimentPie() #Only For deployed version
    langPie()


page_names_to_funcs = {
    "Main Page": mainPage,
    "Data Visualizations": plots,
} 

selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()
