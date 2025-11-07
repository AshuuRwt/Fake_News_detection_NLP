import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize stemmer and vectorizer
port_stem = PorterStemmer()
vector_form = pickle.load(open('vector.pkl', 'rb'))
load_model = pickle.load(open('model.pkl', 'rb'))

# Function to preprocess text
def stemming(content):
    con = re.sub('[^a-zA-Z]', ' ', content)
    con = con.lower().split()
    con = [port_stem.stem(word) for word in con if word not in stopwords.words('english')]
    return ' '.join(con)

# Function to predict fake news
def fake_news(news):
    news = stemming(news)
    input_data = [news]
    vector_form1 = vector_form.transform(input_data)
    prediction = load_model.predict(vector_form1)
    return prediction

# --- Streamlit app ---
st.title('Fake News Classification App')
st.subheader("Input the news content below")
sentence = st.text_area("Enter your news content here", "", height=200)
predict_btt = st.button("Predict")

if predict_btt:
    prediction_class = fake_news(sentence)
    if prediction_class == [0]:
        st.success('Reliable')
    elif prediction_class == [1]:
        st.warning('Unreliable')