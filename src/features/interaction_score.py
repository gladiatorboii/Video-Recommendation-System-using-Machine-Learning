import pandas as pd


def compute_engagement_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute engagement score used as relevance label.
    """
    df = df.copy()

    df["engagement_score"] = (
        0.5 * (df["watch_duration"] / 100.0)
        + 0.2 * df["liked"]
        + 0.2 * df["commented"]
        + 0.1 * df["subscribed_after_watching"]
    )

    return df
