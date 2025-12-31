import pandas as pd


def basic_eda(df: pd.DataFrame) -> None:
    """
    Perform basic EDA and print key insights.
    """

    print("\n===== BASIC DATA OVERVIEW =====")
    print(f"Total interactions: {len(df)}")
    print(f"Unique users: {df['user_id'].nunique()}")
    print(f"Unique videos: {df['video_id'].nunique()}")

    print("\n===== WATCH DURATION =====")
    print(df["watch_duration"].describe())

    print("\n===== BINARY ENGAGEMENT RATES =====")
    print("Like rate:", df["liked"].mean())
    print("Comment rate:", df["commented"].mean())
    print("Subscription rate:", df["subscribed_after_watching"].mean())

    print("\n===== TOP VIDEO CATEGORIES =====")
    print(df["category"].value_counts().head(5))

    print("\n===== TAG STATISTICS =====")
    tag_counts = df["tags"].dropna().apply(lambda x: len(str(x).split(",")))
    print(tag_counts.describe())

    print("\n===== USER ACTIVITY DISTRIBUTION =====")
    print(df.groupby("user_id").size().describe())

    print("\n===== VIDEO INTERACTION DISTRIBUTION =====")
    print(df.groupby("video_id").size().describe())
