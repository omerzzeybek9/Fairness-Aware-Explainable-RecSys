import pandas as pd

GENRE_COLS = [
    "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "FilmNoir", "Horror", "Musical",
    "Mystery", "Romance", "SciFi", "Thriller", "War", "Western",
]

def load_ml100k(data_dir="ml-100k"):
    ratings = pd.read_csv(
        f"{data_dir}/u.data", sep="\t",
        names=["userId", "movieId", "rating", "timestamp"],
    )
    movies = pd.read_csv(
        f"{data_dir}/u.item", sep="|", encoding="latin-1",
        names=[
            "movieId", "title", "release_date", "video_release_date", "imdb_url",
            "unknown", "Action", "Adventure", "Animation", "Children", "Comedy",
            "Crime", "Documentary", "Drama", "Fantasy", "FilmNoir", "Horror",
            "Musical", "Mystery", "Romance", "SciFi", "Thriller", "War", "Western",
        ],
    )
    users = pd.read_csv(
        f"{data_dir}/u.user", sep="|", encoding="latin-1",
        names=["userId", "age", "gender", "occupation", "zip"],
    )
    print(f"Loaded ML-100K -> Ratings: {ratings.shape}, Movies: {movies.shape}, Users: {users.shape}")
    print(f"Gender distribution: {users['gender'].value_counts().to_dict()}")
    return ratings, movies, users


def load_ml1m(data_dir="ml-1m"):
    ratings = pd.read_csv(
        f"{data_dir}/ratings.dat", sep="::", engine="python",
        names=["userId", "movieId", "rating", "timestamp"],
    )
    movies = pd.read_csv(
        f"{data_dir}/movies.dat", sep="::", engine="python",
        encoding="latin-1", names=["movieId", "title", "genres"],
    )
    users = pd.read_csv(
        f"{data_dir}/users.dat", sep="::", engine="python",
        names=["userId", "gender", "age", "occupation", "zip"],
    )
    for g in GENRE_COLS:
        movies[g] = movies["genres"].str.contains(g, case=False).astype(int)

    print(f"Loaded ML-1M -> Ratings: {ratings.shape}, Movies: {movies.shape}, Users: {users.shape}")
    print(f"Gender distribution: {users['gender'].value_counts().to_dict()}")
    return ratings, movies, users


def subset_data(ratings, movies, users,
                max_users=200, max_movies=250, like_threshold=4,
                balance_gender=False):
    user_activity = ratings.groupby("userId").size().sort_values(ascending=False)

    if balance_gender:
        per_gender = max_users // 2
        gender_map = users.set_index("userId")["gender"]
        male_ids = [uid for uid in user_activity.index if gender_map.get(uid) == "M"][:per_gender]
        female_ids = [uid for uid in user_activity.index if gender_map.get(uid) == "F"][:per_gender]
        subset_users = pd.Index(male_ids + female_ids)
    else:
        subset_users = user_activity.head(max_users).index

    ratings_u = ratings[ratings["userId"].isin(subset_users)].copy()
    positive_interactions = ratings_u[ratings_u["rating"] >= like_threshold].copy()

    subset_movies = (
        positive_interactions.groupby("movieId").size()
        .sort_values(ascending=False)
        .head(max_movies).index
    )
    positive_interactions = positive_interactions[
        positive_interactions["movieId"].isin(subset_movies)
    ].copy()

    ratings_sub = ratings_u[ratings_u["movieId"].isin(subset_movies)].copy()
    movies_sub = movies[movies["movieId"].isin(subset_movies)].copy()
    users_sub = users[users["userId"].isin(subset_users)].copy()

    user_info = users_sub.set_index("userId")[["gender", "age", "occupation"]].to_dict("index")

    print(f"Subset -> Users: {len(users_sub)}, Movies: {len(movies_sub)}, "
          f"Positive interactions: {len(positive_interactions)}")
    print(f"Gender split: {users_sub['gender'].value_counts().to_dict()}")

    return ratings_sub, movies_sub, users_sub, positive_interactions, user_info


def train_test_split(positive_interactions, movies_sub, test_ratio=0.2,
                     kg_movie_titles=None):
    movie_id_to_label = dict(zip(movies_sub["movieId"], movies_sub["title"]))
    valid_movie_ids = set(movies_sub["movieId"])

    interactions = positive_interactions[
        positive_interactions["movieId"].isin(valid_movie_ids)
    ].copy().sort_values(["userId", "timestamp"])

    train_list, test_list = [], []
    skipped_users = 0

    for uid, group in interactions.groupby("userId"):
        n = len(group)
        if n < 2:
            skipped_users += 1
            continue
        split_idx = max(1, min(int(n * (1 - test_ratio)), n - 1))
        train_list.append(group.iloc[:split_idx])
        test_list.append(group.iloc[split_idx:])

    if skipped_users > 0:
        print(f"Skipped {skipped_users} users with fewer than 2 interactions.")

    train_interactions = pd.concat(train_list, ignore_index=True)
    test_interactions = pd.concat(test_list, ignore_index=True)

    total = len(train_interactions) + len(test_interactions)
    print(f"Total (after user filter): {total}")
    print(f"Train: {len(train_interactions)} ({len(train_interactions)/total*100:.1f}%)")
    print(f"Test:  {len(test_interactions)} ({len(test_interactions)/total*100:.1f}%)")
    print(f"Avg test items per user: {test_interactions.groupby('userId').size().mean():.1f}")

    user_item_edges = [
        (f"User_{int(uid)}", "likes", movie_id_to_label[int(mid)])
        for uid, mid in train_interactions[["userId", "movieId"]]
        .itertuples(index=False, name=None)
    ]

    test_set_dict = {}
    for row in test_interactions.itertuples(index=False):
        user_node = f"User_{int(row.userId)}"
        movie = movie_id_to_label[int(row.movieId)]
        if kg_movie_titles is not None and movie not in kg_movie_titles:
            continue
        test_set_dict.setdefault(user_node, []).append(movie)

    test_set_dict = {u: mvs for u, mvs in test_set_dict.items() if mvs}

    print(f"\nTrain edges: {len(user_item_edges)}")
    print(f"Test users:  {len(test_set_dict)}")
    print(f"Test pairs:  {sum(len(v) for v in test_set_dict.values())}")
    if kg_movie_titles is not None:
        n_filtered = len(test_interactions) - sum(len(v) for v in test_set_dict.values())
        print(f"Test pairs filtered (not in KG): {n_filtered}")

    return train_interactions, test_interactions, user_item_edges, test_set_dict
