import pandas as pd

def recommend_pearson(movie_name, user_item_matrix, movie_ratings_df, n=5):

    movie_ratings = user_item_matrix[movie_name]

    similar_movies = user_item_matrix.corrwith(movie_ratings)

    corr_df = pd.DataFrame(similar_movies, columns=["PearsonCorr"])

    corr_df.dropna(inplace=True)

    corr_df = corr_df.join(movie_ratings_df.groupby('Title')['Rating'].agg(['count','mean']))

    recommendations = corr_df[corr_df['count'] > 50].sort_values('PearsonCorr', ascending=False)

    return recommendations.head(n)