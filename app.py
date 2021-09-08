import streamlit as st
import pandas as pd
import numpy as np
import os
import seaborn as sns
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
import gensim
from gensim import corpora
from gensim.models import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim
import nltk
from dateutil.parser import parse
import streamlit.components.v1 as components
from datetime import datetime
from datetime import date

@st.cache
def load_data():
    bdf = pd.read_csv('data/biden.csv')
    bdf['newdate'] = pd.to_datetime(bdf['date']).dt.date
    tdf = pd.read_csv('data/trump.csv')
    tdf['newdate'] = pd.to_datetime(tdf['date']).dt.date
    return bdf, tdf

bdf, tdf = load_data()




st.sidebar.title("Campaign Speeches")
page = st.sidebar.radio(
     "Pick an option",
     ('Home' , 'Speech', 'LDA'),
     )

if page == "Home":
    st.header("Overview")

    st.markdown("This page will serve as an overview of all of the analysis that the other pages consist of.")
    st.markdown("-------")
    st.subheader("First a brief overview of the intent:")
    st.markdown("We collect all speech material from both Trump and Biden during the election cycle. For data see the **'Speech'** page.")
    st.markdown("-------")
    st.subheader("Next we implement a natural language processing technique of unsupervised machine learning")
    st.markdown("This is known as Latent Dirichlet Allocation --- Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent dirichlet allocation. Journal of machine Learning research, 3(Jan), 993-1022.")
    st.markdown("This method allows us to identify the underlying topics emergent in the speeches. Find an interactive implementation on the **'LDA'** page.")
    st.markdown("-------")
    st.subheader("Finally we run temporal models using Dynamic Topic modeling")
    st.markdown("This is to better uncover trends in topics over time ---Blei, D. M., & Lafferty, J. D. (2006, June). Dynamic topic models. In Proceedings of the 23rd international conference on Machine learning (pp. 113-120).")
    st.markdown("-------")
    st.header("For more information select a particular topic:")
    extra = st.selectbox("Topic", ("How do you clean speeches?", "What can we change in the LDA?", "How does the dynamic LDA work?"))

    if extra =="How do you clean speeches?":
        st.subheader("Cleaning speeches is a key part in the process.")
        st.markdown("Without cleaning up the data, words and symbols that are not informative might emerge like _'.'_ or _'and'_. Primary steps in no particular order include:")
        st.markdown("* Remove stop words")
        st.markdown("* Tag parts of speech and remove non-nouns")
        st.markdown("* Lemmatize words")
        st.markdown("* Remove symbols and punctuation")

    elif extra == "What can we change in the LDA?":
        st.subheader("A few options are given to be changed in the overall LDA")
        st.markdown("* Number of topics")
        st.markdown("This parameter allows the user to adjust the number of topics the unsupervised machine learning fits the data to. More topics can mean more specific words, but at some point the model may underperform.")
        st.markdown("* Bigram Threshold")
        st.markdown("This parameter changes the number of times two words need to appear together for them to form a bigram, for instance _'United States'_ . Lower numbers indicate fewer times.  ")
        st.markdown("* Date")
        st.markdown("The two inputs correspond to the given date range you wish to constrain a topic model to.")
        st.markdown("* Coherence Score")
        st.markdown("This is a measure of how well the LDA model fits the underlying data. Lower scores here are better. It is the Intrinsic UMass Measure.")

    elif extra == "How does the dynamic LDA work?":
        st.subheader("Dynamic Topic Model Splits the Data")

elif page == "Speech":

    elect = st.radio(
         "Pick an Candidate",
         ('Biden', 'Trump'),
         )
    if elect== "Biden":
        df = bdf
    else:
        df = tdf

    ratin = list(df['date'].unique())
    l = st.selectbox("Date",sorted(ratin))

    x = 0
    for i in range(len(df.loc[df.date == l]['textfull'])):
        t = str(x+1)
        st.subheader("Speech: " + t)
        st.write(df.loc[df.date == l]['textfull'].values[x])

        x+=1

elif page== "LDA":

    st.subheader("LDA Page")

    elect = st.radio(
         "Pick an Candidate",
         ('Biden', 'Trump'),
         )

    if elect== "Biden":
        df = bdf
    else:
        df = tdf



    col1, col2 = st.beta_columns(2)
    with col1:

        numtopics = st.number_input('Number topics', min_value=float(3.0), max_value=float(35.0), value=float(7.0), step=float(1))

        bigram = st.number_input('Bigram Throlshold', min_value=float(1.0), max_value=float(100.0), value=float(50.0), step=float(1))

    #first = dt.datetime.strptime(([datetime.date(2019, 7, 6)]))
    #st.write(first)
    with col2:
        s = st.date_input( "When should we start the LDA", date(2020, 3, 1))

        e = st.date_input( "When should we end the LDA", date(2020, 11, 3))


    mask = (df['newdate'] > s) & (df['newdate'] <= e)
    df = df.loc[mask]


    def preprocess(text):
        result = []
        for token in gensim.utils.simple_preprocess(text):
            if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 2:
                result.append(token)
        return result


    def run_lda(df, savename, numtop=numtopics, thresh=bigram):


        doc_processed = df['final'].map(preprocess)
        dictionary = corpora.Dictionary(doc_processed)
        #doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_processed]

        bigram = gensim.models.Phrases(list(doc_processed), min_count=5, threshold=thresh)
        bigram_mod = gensim.models.phrases.Phraser(bigram)


        def make_bigrams(texts):
            return [bigram_mod[doc] for doc in texts]

        data_words_bigrams = make_bigrams(doc_processed)
        id2word = corpora.Dictionary(data_words_bigrams)

        texts = data_words_bigrams

        corpus = [id2word.doc2bow(text) for text in texts]


        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                               id2word=id2word,
                                               num_topics=numtop,
                                               random_state=100,
                                               update_every=1,
                                               passes=10,
                                               alpha='auto',
                                               per_word_topics=True)


        vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
        save = savename + ".html"
        pyLDAvis.save_html(vis, save)
        return lda_model, doc_processed, dictionary


    save = 'temp.html'

    lda_model, doc_processed, dictionary = run_lda(df, "temp", numtop=numtopics, thresh=bigram)


    coherence_model_lda = CoherenceModel(model=lda_model, texts=doc_processed, dictionary=dictionary, coherence='u_mass')
    coherence_lda = coherence_model_lda.get_coherence()
    st.write("The number of topics is: " + str(numtopics))
    st.write('\nCoherence Score (Lower is better): ', coherence_lda)

    HtmlFile = open('temp.html', 'r', encoding='utf-8')
    source_code = HtmlFile.read()
    components.html(source_code, width=1200, height=1200)
