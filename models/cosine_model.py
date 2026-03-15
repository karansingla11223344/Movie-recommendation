from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


def recommend_cosine(movie_name, user_item_matrix, n=5):
    item_matrix = user_item_matrix.fillna(0).T

    cosine_sim = cosine_similarity(item_matrix)

    cosine_sim_df = pd.DataFrame(
        cosine_sim,
        index=item_matrix.index,
        columns=item_matrix.index
    )

    similar_movies = cosine_sim_df[movie_name].sort_values(ascending=False)[1:n + 1]

    return similar_movies


