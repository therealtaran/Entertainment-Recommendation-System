from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
import numpy as np
import pandas as pd
import ast
import nltk

# %%
movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")
credits.rename(columns={"movie_id": "id"}, inplace=True)

# %%
movies = movies.merge(credits, on="id")


# %%
# movies.columns
# relevant columns
# id
# genres
# keywords
# title
# overview
# cast
# crew
movies = movies[["id", "genres", "keywords",
                 "title_x", "overview", "cast", "crew"]]
# movies.info()

# %%
# now will preprocess the columns
movies.isnull().sum()

# %% [markdown]
#

# %%
movies.dropna(inplace=True)

# %%
movies.duplicated().sum()

# %%
# now we will put sanitize genres


def convert_one(genre_str):
    genre_final = []
    for item in ast.literal_eval(genre_str):
        genre_final.append(item["name"])

    return genre_final

# movies["genres"]=movies["genres"].apply(convert_genres)
# movies["keywords"]=movies["keywords"].apply(convert_genres)


# %%
movies["genres"] = movies["genres"].apply(convert_one)

# %%
movies["keywords"] = movies["keywords"].apply(convert_one)

# %%
# now we will extract only first 3 actors of each movie


def convert_two(cast):
    list_final = []
    for item in ast.literal_eval(cast):
        list_final.append(item["name"])
        if (len(list_final) == 3):
            break

    return list_final

# print(convert_two(movies.iloc[0].cast))


# %%
movies["cast"] = movies["cast"].apply(convert_two)

# %%


def get_director(crew):
    list_final = []
    for c in ast.literal_eval(crew):
        if (c["job"] == "Director"):
            list_final.append(c["name"])
            return list_final

# print(get_director(movies.iloc[0].crew))


# %%
movies["crew"] = movies["crew"].apply(get_director)


# %%
movies["overview"] = movies["overview"].apply(lambda x: x.split())

# %%
movies.head(2)

# %%
movies["genres"] = movies["genres"].apply(
    lambda x: [i.replace(" ", "") for i in x])


# %%
movies["keywords"] = movies["keywords"].apply(
    lambda x: [i.replace(" ", "") for i in x])


# %%
movies["cast"] = movies["cast"].apply(
    lambda x: [i.replace(" ", "") for i in x])


# %%
def convert_three(crew_name):
    list_final = []
    list_final.append(crew_name[0].replace(" ", ""))
    return list_final


# %%
movies.dropna(inplace=True)

# %%
movies["crew"] = movies["crew"].apply(convert_three)

# %%
movies["tags"] = movies["overview"]+movies["genres"] + \
    movies["keywords"]+movies["cast"]+movies["crew"]

# %%
movies.head()

# %%
new_movies = movies[["id", "title_x", "tags"]]

# %%
new_movies["tags"] = new_movies["tags"].apply(lambda x: " ".join(x))

# %%
new_movies["tags"] = new_movies["tags"].apply(lambda x: x.lower())

# %%
new_movies["tags"]

# %%
ps = PorterStemmer()

# %%


def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))

    return " ".join(y)


# %%
new_movies["tags"] = new_movies["tags"].apply(stem)

# %%
cv = CountVectorizer(max_features=5000, stop_words="english")

# %%
vectors = cv.fit_transform(new_movies["tags"]).toarray()

# %%
vectors.shape

# %%

# %%
result = cosine_similarity(vectors)

# %%


def recommend(movie_title):
    movie_index = new_movies[new_movies["title_x"] == movie_title].index[0]
    distances = result[movie_index]
    movies_list = sorted(list(enumerate(distances)),
                         reverse=True, key=lambda x: x[1])[1:6]

    output = []
    for mov in movies_list:
        output.append(new_movies.iloc[mov[0]]["title_x"])
    return output
