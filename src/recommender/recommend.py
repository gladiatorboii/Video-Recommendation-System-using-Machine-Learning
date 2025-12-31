import pandas as pd


def recommend_videos(model, df, X, user_id, top_k=5):
    '''
    Recommend top-k videos for a given user based on predicted engagement.
    '''

    if user_id not in df['user_id'].values:
        raise ValueError(f'User {user_id} not found in dataset')

    # Predict engagement scores
    predictions = model.predict(X)

    df_results = df.copy()
    df_results['predicted_score'] = predictions

    # Videos already watched by the user
    watched_videos = df_results[df_results['user_id'] == user_id]['video_id'].tolist()

    # Candidate videos (not watched)
    candidates = df_results[~df_results['video_id'].isin(watched_videos)]

    # Rank by predicted engagement
    recommendations = (
        candidates
        .sort_values('predicted_score', ascending=False)
        .head(top_k)
    )

    return recommendations[
        ['video_id', 'category', 'tags', 'predicted_score']
    ]
