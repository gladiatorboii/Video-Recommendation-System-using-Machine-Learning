from sklearn.ensemble import RandomForestRegressor


def train_model(X, y):
    """
    Train Random Forest regressor for video engagement prediction.
    Works well on small datasets.
    """

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )

    model.fit(X, y)

    return model
