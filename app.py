import pandas as pd
from fastapi import FastAPI

from models.pearson_model import recommend_pearson
from models.cosine_model import recommend_cosine
from models.svd_model import train_svd
from models.svd_model import recommend_svd


app = FastAPI()

# load data
movies = pd.read_csv('data/movies.csv', sep='::', engine='python', encoding='latin-1')
ratings = pd.read_csv('data/ratings.csv', sep='::', engine='python', encoding='latin-1')
users = pd.read_csv('data/users.csv', sep='::', engine='python', encoding='latin-1')

print("Movies:", movies.shape)
print("Ratings:", ratings.shape)
print("Users:", users.shape)

# fix column name
movies.rename(columns={'Movie ID':'MovieID'}, inplace=True)

# merge datasets
movie_ratings = pd.merge(ratings, movies, on="MovieID")

print("Merged data:", movie_ratings.shape)

# create user-item matrix
user_item_matrix = movie_ratings.pivot_table(
    index="UserID",
    columns="Title",
    values="Rating"
)

print("User-Item Matrix shape:", user_item_matrix.shape)

movie_ratings_df = movie_ratings





@app.get("/pearson")
def pearson_api(movie: str, n: int = 5):

    recs = recommend_pearson(movie, n)

    return recs.reset_index().to_dict(orient="records")



@app.get("/cosine")
def cosine_api(movie: str, n: int = 5):

    recs = recommend_cosine(movie, n)

    return recs.reset_index().to_dict(orient="records")


svd_model = train_svd(ratings)

@app.get("/svd")
def svd_api(user_id: int, n: int = 5):

    recs = recommend_svd(user_id,movies,svd_model, n)

    return recs
