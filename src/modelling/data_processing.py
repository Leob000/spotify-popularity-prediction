import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


class DataFrameTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, drop_cols=True, make_plots=False):
        self.drop_cols = drop_cols
        self.make_plots = make_plots
        # self.index_name = "row_id"
        # self.cols_cat = [
        #     "explicit",
        #     "key",
        #     "mode",
        #     "track_genre",
        #     "time_signature",
        #     "is_time_signature_4",
        #     "is_time_signature_0",
        #     "is_time_signature_1_3_5",
        #     "circle_fifth",
        # ]
        # self.cols_num = [
        #     "popularity",
        #     "duration_ms",
        #     "danceability",
        #     "energy",
        #     "loudness",
        #     "speechiness",
        #     "acousticness",
        #     "instrumentalness",
        #     "liveness",
        #     "valence",
        #     "tempo",
        #     "circle5_sin",
        #     "circle5_cos",
        # ]

    def make_time_signature_cats(self, df_init) -> pd.DataFrame:
        """One-hot encode 'time_signature' into three binary columns and drop the original column."""
        df = df_init.copy()
        df["is_time_signature_4"] = (df["time_signature"] == 4).astype(int)
        df["is_time_signature_0"] = (df["time_signature"] == 0).astype(int)
        df["is_time_signature_1_3_5"] = df["time_signature"].isin([1, 3, 5]).astype(int)
        if self.drop_cols:
            df.drop(columns=["time_signature"], inplace=True)
        return df

    def make_circle_of_fifths(self, df_init) -> pd.DataFrame:
        """Create circle of fifths features from 'key' and 'mode' columns, optionally plotting distributions."""
        df = df_init.copy()

        major_order = [0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5]
        fifths_pos = {k: i for i, k in enumerate(major_order)}

        key = df["key"].astype("Int64")
        mode = df["mode"].astype("Int64")  # 1=major, 0=minor

        # For minor rows, shift by +3 semitones to get the relative major
        k_eff = (key.astype("float") + (1 - mode.astype("float")) * 3) % 12
        df["circle_fifth"] = k_eff.map(fifths_pos).astype("Int64")

        # Feature engineering: sin/cos representation
        theta = 2 * np.pi * df["circle_fifth"] / 12
        df["circle5_sin"] = np.sin(theta)
        df["circle5_cos"] = np.cos(theta)
        if self.make_plots:
            df["circle_fifth"].value_counts().sort_index().plot(kind="bar")
            df.boxplot(column="popularity", by="circle_fifth")
            plt.show()

            plt.scatter(
                df["circle5_sin"], df["circle5_cos"], c=df["popularity"], cmap="viridis"
            )
            plt.colorbar(label="Popularity")
            plt.xlabel("Circle of Fifths (sin)")
            plt.ylabel("Circle of Fifths (cos)")
            plt.title("Circle of Fifths Representation")
            plt.axis("equal")
            plt.show()

        if self.drop_cols:
            df.drop(columns=["key", "circle_fifth"], inplace=True)
        return df

    def scale_popularity(self, df_init):
        df = df_init.copy()
        if "popularity" in df.columns:
            df["popularity"] = df["popularity"] / 100.0
        return df

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        if "row_id" in df.columns:
            df = df.set_index("row_id")
        if df["explicit"].dtype != int:
            df["explicit"] = df["explicit"].astype(int)
        df = self.make_time_signature_cats(df)
        df = self.make_circle_of_fifths(df)
        df = self.scale_popularity(df)
        return df

    def set_output(self, *, transform=None):  # type:ignore
        # For compatibility with sklearn Pipeline
        self._output_config = {"transform": transform}
        return self


class CustomColumnScaler(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        extend_standard_scaling: bool = False,
        output_as_pandas: bool = True,
    ):
        self.extend_standard_scaling = extend_standard_scaling
        self.output_as_pandas = output_as_pandas

        self.base_cols = [
            "duration_ms",
            "tempo",
            "loudness",
        ]
        self.extended_cols = [
            "danceability",
            "energy",
            "speechiness",
            "acousticness",
            "instrumentalness",
            "liveness",
            "valence",
        ]

    def fit(self, X, y=None):
        cols_to_normalize = self.base_cols.copy()
        if self.extend_standard_scaling:
            cols_to_normalize += self.extended_cols
        self.cols_to_normalize_ = [col for col in cols_to_normalize if col in X.columns]
        self.scaler_ = StandardScaler()
        self.scaler_.fit(X[self.cols_to_normalize_])
        self.other_cols_ = [
            col for col in X.columns if col not in self.cols_to_normalize_
        ]
        return self

    def transform(self, X):
        X_scaled = X.copy()
        X_scaled[self.cols_to_normalize_] = self.scaler_.transform(
            X[self.cols_to_normalize_]
        )
        result = X_scaled
        if self.output_as_pandas:
            return pd.DataFrame(result, index=X.index, columns=result.columns)
        else:
            return result.values
