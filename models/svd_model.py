from surprise import Dataset
from surprise import Reader
from surprise import SVD

def train_svd(ratings):

    reader = Reader(rating_scale=(1,5))

    data = Dataset.load_from_df(
        ratings[['UserID','MovieID','Rating']],
        reader
    )

    trainset = data.build_full_trainset()

    model = SVD()

    model.fit(trainset)

    return model

def recommend_svd(user_id,movies,model,n=5):

    movies_list = movies['MovieID'].unique()

    predictions = []

    for movie_id in movies_list:

        pred = model.predict(user_id, movie_id)

        predictions.append((movie_id, pred.est))

    predictions.sort(key=lambda x: x[1], reverse=True)

    top_movies = predictions[:n]

    results = []

    for movie_id, rating in top_movies:

        title = movies[movies['MovieID']==movie_id]['Title'].values[0]

        results.append({"title": title, "rating": rating})

    return results
