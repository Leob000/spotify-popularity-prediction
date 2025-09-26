# %%
# ruff: noqa: E402
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Util to import functions from project root
p = Path.cwd()
root = next(
    (
        parent
        for parent in [p] + list(p.parents)
        if (parent / "pyproject.toml").exists()
    ),
    None,
)
if root is None:
    root = Path.cwd()
sys.path.insert(0, str(root))

from src.modelling.data_processing import DataFrameTransformer

# %%
df = pd.read_csv("../src/data/train_data.csv")
cols = df.columns.tolist()
df.head()
# %%
df.set_index("row_id", inplace=True)
df["explicit"] = df["explicit"].astype(int)
df.describe()
# %%
# Check missing values
assert (df.isnull().sum() == 0).all()

cols_cat = [
    "explicit",
    "key",
    "mode",
    "track_genre",
    "time_signature",
]
cols_num = [
    "popularity",
    "duration_ms",
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
]

df.dtypes

# %%
print(df["time_signature"].value_counts())

for col in cols_cat:
    # plt.hist(df[col], bins=df[col].nunique())  # type:ignore
    df[col].value_counts().sort_index().plot(kind="bar")
    plt.show()

# time_signature: 4 is super common, 3-1-5 rare, 0 very rare.

# %%
print("Number of genres:", len(df["track_genre"].unique()))
df["track_genre"].value_counts().hist()
plt.show()

# The dataset seems somewhat balanced in terms of genres, every genre has around 750 observations.
# The genre seems to bring of lot of information to predict popularity.
# There are 114 genres, we could transform each in a categorical variable, but it could be inefficient for some class of models.

# An idea would be, instead of predicting the popularity directly, to estimate the mean popularity by genre,
# and to predict, for a sample of genre X, the residuals (popularity - estimated_mean_popularity_genre_X).
# The final prediction would be the sum of the estimated mean popularity by genre and the predicted residuals.
# (We could even try ensemble learning, by training this model, and a more "classical" model).

mean_pop_by_genre = df.groupby("track_genre")["popularity"].mean().sort_values()  # type:ignore
plt.figure(figsize=(15, 5))
mean_pop_by_genre.plot(kind="bar")
plt.ylabel("Mean Popularity")
plt.xlabel("Genre")
plt.title("Mean Popularity by Genre")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# %%
for col in cols_cat:
    df.boxplot(column="popularity", by=col)
    plt.title("")
    plt.suptitle("")
    plt.xlabel(f"{col}")
    plt.ylabel("Popularity")
    plt.show()

# time_signature: 1/2/5 seem quite similar, 4 is the most common and is different, 0 is rare and different.
# -> We could group the signatures by 3 categorical variable: is4, is0, is1-3-5

df_transformer = DataFrameTransformer(drop_cols=True, make_plots=True)
df = df_transformer.make_time_signature_cats(df)

# %%
# The key and the mode doesn't seem to capture a lot of popularity information.
# Thanks to domain knowledge, we can engineer a new feature: the position in the circle of fifths.
# See: https://en.wikipedia.org/wiki/Circle_of_fifths#/media/File:Circle_of_fifths_deluxe_4.svg
# We get 12 categorical variables, we can either keep them as this or represent them with sin/cos.

df = df_transformer.make_circle_of_fifths(df)

# %%
for col in cols_num:
    if col != "popularity":
        plt.scatter(df[col], df["popularity"], alpha=0.1, s=5)
        plt.xlabel(col)
        plt.ylabel("Popularity")
        plt.title(f"Popularity vs {col}")
        plt.show()

# Identification of some outliers and clusters, we could study further to see if they correspond to specific genres.

# %%
# Test the pipeline
from src.modelling.data_processing import CustomColumnScaler

df_transformer = DataFrameTransformer()
scaler = CustomColumnScaler()


X_train = pd.read_csv("../src/data/train_data.csv")
X_train = df_transformer.fit_transform(X_train)
X_train = scaler.fit_transform(X_train)
X_train.describe()

# %%
