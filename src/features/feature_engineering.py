import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer


def prepare_features(df: pd.DataFrame):
    """
    Prepare features and integer relevance labels for LambdaRank.
    """

    df = df.copy()

    # FORCE integer relevance labels
    df["relevance"] = pd.cut(
        df["engagement_score"],
        bins=[-1, 0.25, 0.5, 0.75, 1.0],
        labels=[0, 1, 2, 3]
    ).astype(int)

    # Encode categorical features
    user_encoder = LabelEncoder()
    video_encoder = LabelEncoder()
    category_encoder = LabelEncoder()

    df["user_enc"] = user_encoder.fit_transform(df["user_id"])
    df["video_enc"] = video_encoder.fit_transform(df["video_id"])
    df["category_enc"] = category_encoder.fit_transform(df["category"])

    # TF-IDF for tags
    tfidf = TfidfVectorizer()
    tag_features = tfidf.fit_transform(df["tags"])

    tag_df = pd.DataFrame(
        tag_features.toarray(),
        columns=[f"tag_{i}" for i in range(tag_features.shape[1])]
    )

    # Numeric features
    numeric_df = df[
        ["watch_duration", "liked", "commented", "subscribed_after_watching"]
    ].reset_index(drop=True)

    # Final feature matrix
    X = pd.concat(
        [
            df[["user_enc", "video_enc", "category_enc"]].reset_index(drop=True),
            numeric_df,
            tag_df.reset_index(drop=True),
        ],
        axis=1,
    )

    # FINAL LABEL (INT ONLY)
    y = df["relevance"].astype(int)

    # DEBUG 
    print("\n[DEBUG] Unique relevance labels:", y.unique())
    print("[DEBUG] Label dtype:", y.dtype)

    # Group by user (LambdaRank requirement)
    group = df.groupby("user_id").size().tolist()

    return X, y

