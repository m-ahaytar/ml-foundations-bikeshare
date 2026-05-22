![Bike Share Header](bike_share_header.png)

# Chicago Divvy Bikeshare — ML Project

## Project Overview

This project applies a range of machine learning models to the Chicago Divvy bikeshare dataset (July 2023). The goal is to predict rider type (member vs casual), estimate trip duration, and find patterns in trip behavior using clustering and PCA.

It was built as a hands-on machine learning course project. The notebook goes through the full workflow: loading the data, cleaning it, exploring patterns, training multiple models, and interpreting the results.

## Objectives

- Predict whether a rider is a member or casual rider (classification)
- Predict trip duration from trip features (regression)
- Find natural groupings of trips using clustering
- Reduce dimensionality with PCA and interpret the components
- Practice different evaluation techniques and model comparison

## Dataset

The dataset comes from the Divvy bike-share system in Chicago. It is stored in parquet format.

Key columns used:

- `member_casual` — target for classification
- `trip_duration_min` — target for regression
- `rideable_type` — classic or electric bike
- `started_at` / `ended_at` — timestamps
- `start_lat`, `start_lng`, `end_lat`, `end_lng` — coordinates
- Engineered features: `hour`, `day_of_week`, `is_weekend`, `rush_hour`, `trip_distance_km`

## Project Workflow

1. **Setup** — imports, random seed, library versions
2. **Data loading** — parquet file, shape, missing values
3. **Cleaning** — datetime parsing, trip duration, distance, filtering outliers
4. **EDA** — rider balance, bike type usage, hourly demand, heatmaps, pairplot
5. **Feature engineering** — classification and regression prep with encoding and scaling
6. **Classification** — 10 models with evaluation and tuning
7. **Regression** — 6 models with residual analysis
8. **Clustering** — K-Means, hierarchical, DBSCAN
9. **PCA** — explained variance, loadings, scatter plot
10. **Summary** — takeaways and limitations

## Feature Engineering

- Converted timestamps to hour and day of week
- Computed trip distance using haversine formula
- Created `is_weekend` and `rush_hour` binary flags
- One-hot encoded categorical variables (`rideable_type`, `day_of_week`)
- Standard scaled numeric features
- Log-transformed trip duration for regression
- Separate preprocessing pipelines for classification and regression

## Classification Models

- Dummy Classifier (baseline)
- Logistic Regression
- k-Nearest Neighbors (tuned over k values)
- Support Vector Machine (RBF kernel)
- Naive Bayes (Gaussian)
- Ridge Classifier
- Decision Tree
- Random Forest
- GridSearchCV-tuned Random Forest
- Voting Ensemble (Logistic Regression + Decision Tree + k-NN)
- AdaBoost

Each model includes accuracy, F1-score, precision, recall. The best model includes a confusion matrix, ROC curve, and classification report.

## Regression Models

- Linear Regression
- Ridge Regression
- Lasso Regression
- Elastic-Net
- Random Forest Regressor
- Dummy Regressor (baseline)

All evaluated with MAE, RMSE, and R². Residual plots for the best model.

## Clustering and PCA

- K-Means with elbow method and silhouette score
- Hierarchical clustering with dendrogram visualization
- DBSCAN density-based clustering
- PCA for dimensionality reduction and visualization
- Explained variance plot and component loadings

## Evaluation Metrics

- Classification: accuracy, precision, recall, F1, ROC-AUC, confusion matrix
- Regression: MAE, RMSE, R²
- Clustering: silhouette score, inertia
- Cross-validation and learning curves

## Key Insights

- Random Forest performed best overall for both classification and regression
- Rush hour and trip distance were useful features for distinguishing member vs casual riders
- Trip duration has a long tail — log transformation helped regression models
- K-Means with 3–4 clusters gave meaningful groupings of trip patterns
- PCA showed that 2–3 components capture most of the variance in the feature set

## Technologies Used

- Python
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- pyarrow (parquet)
- scipy

## How to Run

1. Make sure Python 3.9+ is installed with the packages listed above
2. Place `202307-divvy-tripdata.parquet` in the same folder as the notebook
3. Place `bike_share_header.png` in the same folder (used in the title cell)
4. Open `bikeshare_notebook.ipynb` in Jupyter
5. Run all cells top to bottom

## Project Structure

```
.
├── README.md
├── bikeshare_notebook.ipynb
├── 202307-divvy-tripdata.parquet
└── bike_share_header.png
```

## Future Improvements

- Try gradient boosting (XGBoost, LightGBM) for both classification and regression
- Add more weather or calendar features
- Use the full dataset instead of a 20k sample
- Deploy the best model as a simple web app or API

## Author

Ahaytar Mohamed — built as part of a machine learning foundations class.
