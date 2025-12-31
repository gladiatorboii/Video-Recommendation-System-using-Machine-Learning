REQUIRED_COLUMNS = {
    "user_id",
    "video_id",
    "category",
    "tags",
    "watch_duration",
    "liked",
    "commented",
    "subscribed_after_watching",
    "timestamp",
}


def validate_dataset(df):
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if df.isnull().all(axis=1).any():
        raise ValueError("Dataset contains completely empty rows")

    if not df["watch_duration"].between(0, 100).all():
        raise ValueError("watch_duration must be between 0 and 100")

    print("Dataset validation successful")
