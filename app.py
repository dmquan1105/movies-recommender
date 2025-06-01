import streamlit as st
import pandas as pd
import numpy as np
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma 
from langchain_huggingface import HuggingFaceEmbeddings
import os
os.environ["STREAMLIT_WATCH_FILE"] = "false"

@st.cache_data
def load_movies():
    movies_df = pd.read_csv('movies.csv')
    movies_df['Poster_Link'] = movies_df['Poster_Link']
    movies_df['Poster_Link'] = np.where(
        movies_df['Poster_Link'].isna(),
        'poster_not_avail.png',
        movies_df['Poster_Link'],
    )
    return movies_df

@st.cache_resource
def load_db():
    raw_docs = TextLoader('tag_desc.txt').load()
    text_splitter = CharacterTextSplitter(separator='\n', chunk_size=0, chunk_overlap=0)
    docs = text_splitter.split_documents(raw_docs)
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db_movies = Chroma.from_documents(docs, embeddings, persist_directory='db')
    return db_movies

def retrieve_movies(db_movies: Chroma, movies: pd.DataFrame, query: str, top_k:int = 10) -> pd.DataFrame:
    """Retrieve movies based on a query

    Args:
        db_movies (Chroma): The vector store containing movie embeddings.
        movies (pd.DataFrame): The DataFrame containing movie metadata.
        query (str): The query to search for movies.
        top_k (int, optional): The number of top recommendations to return. Defaults to 10.

    Returns:
        pd.DataFrame: A DataFrame containing the recommended movies.
    """
    recs = db_movies.similarity_search_with_score(query, k=50)
    recs = sorted(recs, key=lambda x: x[1], reverse=True)
    movies_list = []
    for i in range(0, len(recs)):
        movies_list += [int(recs[i][0].page_content.strip('""').split()[0])]
    return movies[(movies['id']).isin(movies_list)].head(top_k)

import requests

def is_valid_image(url):
    try:
        response = requests.head(url, timeout=2)
        content_type = response.headers.get('Content-Type', '')
        return response.status_code == 200 and 'image' in content_type
    except:
        return False

movies_data = load_movies()
db_movies_vectorstore = load_db()

st.title("Movie Recommendation System")
st.sidebar.header("Filter Options")

# User input for search query
search_query = st.sidebar.text_input("Search for a movie", "", placeholder="E.g: Superheroes movies")

genres = ['Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 'Drama', 'Family', 'Fantasy', 'Film-Noir', 'History', 'Horror', 'Music', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Sport', 'Thriller', 'War', 'Western']

# Dropdown for genre selection
selected_genres = st.sidebar.selectbox("Filter by Genre", ['All'] + list(genres))

filtered_movies = movies_data.copy()

# Dropdown Imdb rating filter with 4 categories
selected_rating = st.sidebar.selectbox(
    "Filter by IMDB Rating",
    ['All', '0.0 - 3.9', '4.0 - 5.9', '6.0 - 6.9', '7.0 - 8.4', '8.5 - 10.0']
)

# Dropdown duration filter with 4 categories
selected_duration = st.sidebar.selectbox(
    "Filter by Duration",
    ['All', '0 - 60 minutes', '61 - 120 minutes', '121 - 180 minutes', '181+ minutes']
)

if search_query:
    filtered_movies = retrieve_movies(db_movies_vectorstore, filtered_movies, search_query)
    
if selected_genres != 'All':
    filtered_movies = filtered_movies[filtered_movies[f"Genre_{selected_genres}"] == 1]
    
if selected_rating != 'All':
    min_rating, max_rating = map(float, selected_rating.split(' - '))
    filtered_movies = filtered_movies[
        (filtered_movies['IMDB_Rating'] >= min_rating) & 
        (filtered_movies['IMDB_Rating'] <= max_rating)
    ]
    
if selected_duration != 'All':
    if selected_duration == '0 - 60 minutes':
        filtered_movies = filtered_movies[filtered_movies['Runtime'] <= 60]
    elif selected_duration == '61 - 120 minutes':
        filtered_movies = filtered_movies[(filtered_movies['Runtime'] > 60) & (filtered_movies['Runtime'] <= 120)]
    elif selected_duration == '121 - 180 minutes':
        filtered_movies = filtered_movies[(filtered_movies['Runtime'] > 120) & (filtered_movies['Runtime'] <= 180)]
    elif selected_duration == '181+ minutes':
        filtered_movies = filtered_movies[filtered_movies['Runtime'] > 180]
    
st.subheader("Recommended Movies")
if not filtered_movies.empty:
    for _, row in filtered_movies.iterrows():
        poster_url = row['Poster_Link'] if is_valid_image(row['Poster_Link']) else 'poster_not_avail.png'
        st.image(poster_url, width=150)
        st.markdown(
            f"**{row['Series_Title']}** ({int(row['Released_Year'])})  \n"
            f"⭐ **IMDB Rating**: {row['IMDB_Rating']}  |  🎭 **Genre**: {row['Genre']}  |  🎬 **Director**: {row['Director']}  \n"
            f"⏳ **Duration**: {row['Runtime']} minutes  \n"
            f"📖 **Overview**: {row['Overview']}"
        )
        st.markdown("---")
else:
    st.write("No movies found matching your criteria.")