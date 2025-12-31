#FINAL REPORT


# Video Recommendation System using Machine Learning

## 1. Introduction
This project presents a machine learning–based video recommendation system that personalizes content using historical user engagement data.

## 2. Dataset Description
The dataset consists of user–video interactions including behavioral signals such as watch duration, likes, comments, and subscriptions, along with content metadata like tags and categories.

## 3. Exploratory Data Analysis
EDA shows that watch duration and likes are the most influential engagement signals, while comments and subscriptions represent stronger but less frequent user intent.

## 4. Engagement Modeling
An engagement score was computed using implicit feedback signals. This score represents the level of user interest in a video and serves as the target variable.

## 5. Feature Engineering
- Numerical: watch duration
- Binary: liked, commented, subscribed
- Categorical: user ID, video ID, category
- Text: tags using TF-IDF vectorization

## 6. Machine Learning Model
A Random Forest regression model was trained to predict engagement scores. The model captures non-linear relationships and provides interpretable feature importance.

## 7. Recommendation Strategy
For a given user, engagement scores are predicted for unseen videos. Videos are ranked by predicted engagement and the top-K videos are recommended.

## 8. Results
The model learned that watch duration and likes are the strongest predictors of engagement. Personalized recommendations were successfully generated.

## 9. Limitations and Future Work
Due to limited data, timestamp-based temporal modeling was not used. Future work includes time-aware models and larger datasets.

## 10. Conclusion
The project demonstrates a complete end-to-end machine learning pipeline for personalized video recommendation using historical engagement signals.
