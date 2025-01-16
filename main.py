import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# Load your movie dataset
def load_data():
    movies_df = pd.read_csv('movies.csv')  # Replace with your dataset file
    ratings_df = pd.read_csv('ratings.csv')  # Replace with your dataset file
    return movies_df, ratings_df


# Movie Recommender
def movie_recommendations(movie_name, cosine_sim, movies_df):
    idx = movies_df[movies_df['title'] == movie_name].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return movies_df['title'].iloc[movie_indices]


# Title Input and Recommendations
def recommend_movie():
    st.title('Movie Recommendation System')

    movies_df, ratings_df = load_data()

    # Choose a movie
    movie_name = st.selectbox('Select a Movie', movies_df['title'].values)

    # Prepare movie recommendations
    st.write("Recommendations based on your choice:")

    # Simulate cosine similarity
    movie_cosine_sim = cosine_similarity(ratings_df)  # Adjust based on how you are calculating cosine similarity

    recommendations = movie_recommendations(movie_name, movie_cosine_sim, movies_df)

    for movie in recommendations:
        st.write(movie)


# Run the Streamlit app
if __name__ == "__main__":
    recommend_movie()
