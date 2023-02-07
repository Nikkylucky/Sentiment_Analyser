# sentiment analysis app using streamlit

import streamlit as st
import pandas as pd
import numpy as np
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import requests
import pickle
from stop_words import get_stop_words
import joblib

stop_words = get_stop_words('english')
stopwords = set(stop_words)

def test_message(message):
    message = message.replace("@", " ")
    message = message.replace("#", " ")
    message = message.replace("RT", " ")
    message = message.replace(":", " ") 
    message = message.replace(";", " ")
    message = message.replace("!", " ")
    message = message.replace("?", " ")
    message = message.replace(",", " ")
    message = message.replace(".", " ")
    message = str(message)
    message = message.lower() # convert all the text to lower case
    message = message.replace('[^a-zA-Z]', ' ') # remove special characters and numbers from the text
    message = message.split() # split the text into a list of words
    message = [word for word in message if not word in stopwords] # remove stopwords from the list of words 
    message = ' '.join(message) # join the list of words into a sentence
    message = [message] # convert the sentence into a list
    message = cv.transform(message).toarray() # convert the list into an array
    return model.predict(message) # predict the sentiment of the text

model = joblib.load('./finalized_Sentiment_Model.sav')
cv = pickle.load(open('./vectorizer.pkl', 'rb'))

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def main():

    st.set_page_config(page_title="Sentiment Analyzer", page_icon="mailbox_with_mail", layout="wide", initial_sidebar_state="expanded")

    option = option_menu(
        menu_title = None,
        options = ["Home", "Detector"],
        icons = ["house", "gear"],
        menu_icon = 'cast',
        default_index = 0,
        orientation = "horizontal"
    )

    if option == "Home":
        st.markdown("<h1 style='text-align: center; color: #08e08a;'>Sentiment Analyzer</h1>", unsafe_allow_html=True)
        lottie_hello = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_sz1JuRzT9G.json")

# rgb color code for 
        st_lottie(
            lottie_hello,
            speed=1,
            reverse=False,
            loop=True,
            quality="High",
            #renderer="svg",
            height=400,
            width=-900,
            key=None,
        )

        st.markdown("<h4 style='text-align: center; color: #dae5e0;'>Enter your text and let the model decide it's polarity</h4>", unsafe_allow_html=True)

    elif option == "Detector":
        st.markdown("<h1 style='text-align: center; color: #08e08a;'>Sentiment Analyser</h1>", unsafe_allow_html=True)
        text = st.text_area("Enter your text here")

        if st.button("Detect"):
            # check if the text is empty
            if text == "":
                st.error("Please enter some text")
            else:
                # predict the text
                prediction = test_message(text)
                if prediction == 0:
                    st.error("Ooooo, It's Negative Vibe")
                elif prediction == 1:
                    st.success("Zzzzz, It's Neutral Vibe")
                else:
                    st.success("Yayyy, It's Positive Vibe")


if __name__ == "__main__":
    main()
