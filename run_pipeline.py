from src.data.load_data import load_dataset
from src.data.validation import validate_dataset
from src.utils.eda import basic_eda
from src.features.interaction_score import compute_engagement_score
from src.features.feature_engineering import prepare_features
from src.models.train_random_forest import train_model
from src.recommender.recommend import recommend_videos


DATA_PATH = "data/MLAssignmen-VideoRecommendation-Dataset.csv"


def main():
    print("Loading dataset...")
    df = load_dataset(DATA_PATH)

    print("Validating dataset...")
    validate_dataset(df)

    print("Running EDA...")
    basic_eda(df)

    print("\nComputing engagement score...")
    df = compute_engagement_score(df)

    print("\nPreparing features...")
    X, y = prepare_features(df)

    print("\nTraining Random Forest regression model...")
    model = train_model(X, y)

    print("\nModel trained successfully")
    print("Top feature importances:")

    for name, imp in sorted( 
        zip(X.columns, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True
    )[:10]:
        print(f"{name}: {imp:.4f}")

    user_id = input("\nEnter user_id to get recommendations: ").strip()

    try:
        recs = recommend_videos(model, df, X, user_id=user_id, top_k=3)
        print(f"\nTop recommendations for user {user_id}:")
        print(recs)
    except ValueError as e:
        print(e)




if __name__ == "__main__":
    main()
