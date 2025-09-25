import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def make_time_signature_cats(df_init) -> pd.DataFrame:
    """One-hot encode 'time_signature' into three binary columns and drop the original column."""
    df = df_init.copy()
    df["is_time_signature_4"] = (df["time_signature"] == 4).astype(int)
    df["is_time_signature_0"] = (df["time_signature"] == 0).astype(int)
    df["is_time_signature_1_3_5"] = df["time_signature"].isin([1, 3, 5]).astype(int)
    df.drop(columns=["time_signature"], inplace=True)
    return df


def make_circle_of_fifths(df_init, make_plots=False) -> pd.DataFrame:
    """Create circle of fifths features from 'key' and 'mode' columns, optionally plotting distributions."""
    df = df_init.copy()

    major_order = [0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5]
    fifths_pos = {k: i for i, k in enumerate(major_order)}

    key = df["key"].astype("Int64")
    mode = df["mode"].astype("Int64")  # 1=major, 0=minor

    # For minor rows, shift by +3 semitones to get the relative major
    k_eff = (key.astype("float") + (1 - mode.astype("float")) * 3) % 12
    df["circle_fifth"] = k_eff.map(fifths_pos).astype("Int64")
    df["circle_fifth"].value_counts().sort_index().plot(kind="bar")

    # Feature engineering: sin/cos representation
    theta = 2 * np.pi * df["circle_fifth"] / 12
    df["circle5_sin"] = np.sin(theta)
    df["circle5_cos"] = np.cos(theta)
    if make_plots:
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

    df.drop(columns=["key", "mode", "circle_fifth"], inplace=True)
    return df


def transform_df(df_init):
    """Transformations used in the exploration notebook to prepare the dataframe"""
    df: pd.DataFrame = df_init.copy()
    df.set_index("row_id", inplace=True)
    df["explicit"] = df["explicit"].astype(int)

    df = make_time_signature_cats(df)
    df = make_circle_of_fifths(df)
    return df
