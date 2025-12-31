# Video Recommendation System using Machine Learning

## Overview
This project implements a machine learning–based video recommendation system that recommends videos to users based on historical engagement signals such as watch duration, likes, comments, subscriptions, tags, and categories.

## Dataset
The dataset contains user–video interaction records with the following fields:
- user_id
- video_id
- watch_duration
- liked
- commented
- subscribed_after_watching
- category
- tags
- timestamp

## Approach
1. Data loading and validation
2. Exploratory Data Analysis (EDA)
3. Engagement score computation using implicit feedback
4. Feature engineering (numerical, categorical, text)
5. Model training using Random Forest
6. Video recommendation by ranking predicted engagement

## Model Used
- RandomForestRegressor (scikit-learn)
- Chosen for robustness on small datasets and interpretability

## How to Run
```bash
pip install -r requirements.txt
python run_pipeline.py
