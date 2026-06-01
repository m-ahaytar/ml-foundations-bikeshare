![Bike Share Header](bike_share_header.png)

# Chicago Divvy Bikeshare — ML Foundations Project

A machine learning project working through classification, regression, clustering, and PCA on real bikeshare trip data from Chicago's Divvy system.

## The Dataset

Divvy is Chicago's public bike-share system. This project uses the July 2023 trip data from the public Divvy dataset — about 767k raw trips, 746k after cleaning, with 19 columns. For modeling I worked with a 20k-row stratified sample to keep training times manageable.

The key columns are `member_casual` (member or casual rider — the classification target), `rideable_type` (classic or electric bike), start and end timestamps, and GPS coordinates. From these I engineered `trip_duration_min`, `trip_distance_km` (via haversine), `hour`, `day_of_week`, `is_weekend`, and `rush_hour`.

The class split is 58.3% member and 41.7% casual — mildly imbalanced, which matters for how you read accuracy.

![Class Balance](assets/class_balance.png)
*Members outnumber casual riders by about 3-to-2 in this July sample.*

## What We're Trying to Do

**Classification.** Given trip features (bike type, time of day, coordinates, distance), can we predict whether the rider is a member or a casual user? Members have annual passes and tend to use the system differently — more commuting, shorter trips.

**Regression.** Can we estimate trip duration from the same trip features? This is harder since duration depends on things not in the data (traffic, rider fitness, route choice).

**Clustering and PCA.** Without using the rider-type label, do trips naturally form meaningful groups? And can PCA reduce the feature set into something we can wrap our heads around?

## Data Prep & Feature Engineering

The raw data had some issues. A few hundred rows had null end coordinates — dropped. Some trips had negative durations (clock drift) — dropped. I capped duration at 120 minutes and distance at 50 km to keep the models focused on realistic trips.

The features I created:
- **trip_distance_km**: straight-line distance from start to end coordinates using the haversine formula. It's an underestimate of actual road distance but it's the best we can do without route data.
- **hour** and **day_of_week**: extracted from the started_at timestamp. Bike usage shifts dramatically by time of day.
- **is_weekend** and **rush_hour**: binary flags for behavioral context. Rush hour covers 7–9 AM and 4–6 PM on weekdays.
- **log-transformed duration**: the raw trip duration has a long right tail. Linear models can't handle this well, so I applied log1p for regression.

I built separate preprocessing pipelines for classification and regression. The classification pipeline consciously excludes trip duration — using it would be data leakage since you don't know the duration until the ride is over.

## Classification — Who's Riding?

I tested 10 classifiers plus a dummy baseline. The goal was to predict member vs casual from trip features alone — no duration, no rider ID, just what you'd know at trip start.

![Classification Comparison](assets/cls_comparison.png)
*Tuned Random Forest edges out as the best classifier. The Dummy baseline bars are all models to the left of it.*

**Dummy Classifier** (baseline): always predicts the majority class (member). 58.17% accuracy — the minimum bar.

**Manual Rule Classifier**: I wrote a set of if-else rules by hand — "if weekend and long distance, predict casual", "if rush hour, predict member". It scored 57.55%, which is below the dummy baseline. That was the first sign that the boundary between member and casual isn't something you can capture with simple heuristics.

**Logistic Regression**: fits a linear decision boundary. Fast and interpretable, but this data isn't linearly separable. 58.83%.

**k-NN (k=15)**: votes by the nearest training examples. Sensitive to feature scaling (hence the pipeline). Best k was 15 after tuning — higher than the default, suggesting decision boundaries are smooth. 59.60%.

**SVM (RBF kernel)**: this surprised me a bit — 62.05%, the second-best score. The RBF kernel lets SVM draw non-linear boundaries without manually engineering interaction features. I think it worked well here because the data has a mix of categorical and continuous features that aren't linearly separable, exactly the kind of setup where a kernel method shines.

**Naive Bayes**: assumes features are independent given the class. They aren't (distance and duration correlate), which dragged it to 57.85%.

**Ridge Classifier**: regularized linear model, similar to logistic regression. 58.85%.

**Decision Tree (max_depth=8)**: the train-test gap is the interesting thing here — 65.77% train vs 61.52% test. A single tree memorizes patterns that don't generalize, even with depth capped at 8. You can extract the actual rules from it, which is useful for understanding, but I wouldn't trust it for prediction on new data.

**Random Forest (100 trees)**: averages many trees to reduce overfitting. 62.38%. Train accuracy was 79.17% — still some gap, but better than a single tree.

**Tuned RF (GridSearchCV)**: I ran a grid search over n_estimators, max_depth, and min_samples_leaf — the best model hit 62.40% vs the untuned RF's 62.38%. That 0.02% gain for several minutes of compute is honestly underwhelming. scikit-learn's defaults were already almost optimal for this data.

**Voting Ensemble** (LR + DT + k-NN): hard voting across three model types. 60.75%.

**AdaBoost (50 estimators)**: builds trees sequentially, each correcting the previous one's mistakes. 60.80%.

The tuned Random Forest wins, but 62.4% is not a dramatic improvement over the baseline. Members and casual riders don't separate cleanly on trip features alone — especially in July, when both groups take leisure trips.

![Confusion Matrix](assets/confusion_matrix.png)
*The model gets most members right (2089 out of 2327) but struggles badly with casual riders — it predicted 1266 of them as members. When in doubt, it defaults to the majority class.*

## Regression — How Long Will the Trip Take?

For regression I switched targets to trip duration (log-transformed). This turned out to be the harder problem.

![Regression Comparison](assets/reg_comparison.png)
*Only Random Forest scored above the dummy baseline. All linear models were negative.*

**Dummy Regressor**: always predicts the mean. R² = 0.0 — by definition.

**Linear Regression**: R² = -0.877. This is worse than predicting the mean. The features don't have a linear relationship with duration.

**Ridge, Lasso, Elastic-Net**: regularized linear models with slightly different penalties. R² values of -0.880, -0.851, and -0.854. Regularization didn't help because the problem isn't coefficient magnitude — it's that the relationship itself isn't linear.

**Random Forest Regressor**: R² = 0.281, RMSE = 13.98 minutes. The only model that learned anything useful. It captures non-linear interactions between distance, time of day, and location.

Negative R² means the model is actively worse than guessing the average. If you always predicted 15.4 minutes (the mean), you'd be off by 16.5 minutes RMSE. The linear models somehow manage to be off by 22+ minutes. The problem is that short and long trips share the same feature values — a 5-minute trip and a 50-minute trip can have the same distance and start location if one rider bikes fast and the other takes the scenic route.

![Residual Plot](assets/residual_plot.png)
*Random Forest residuals are centered around zero but spread increases for longer predicted durations — the model is less certain about longer trips.*

## Clustering — Finding Trip Patterns

I removed the labels and asked: do trips form natural groups?

**K-Means**: partitions data into k clusters by minimizing within-cluster distances. I tested k=2 through k=8 using the elbow method and silhouette score.

![Elbow and Silhouette](assets/elbow_silhouette.png)
*No sharp elbow — the silhouette peaks at k=7 but values are low across the board.*

The best silhouette score was 0.185 at k=7. That's a weak structure — the data doesn't want to cluster into tight balls.

**Hierarchical clustering**: builds a merge tree bottom-up. I tested cuts at k=2, 3, and 4. Best silhouette was 0.182 at k=3 — consistent with K-Means in that clusters aren't well-separated.

![Dendrogram](assets/dendrogram.png)
*No large jump in merge distance, confirming the data is fairly continuous.*

**DBSCAN**: density-based. It found 2 clusters with 176 noise points from 15,000 trips. Silhouette on non-noise points was 0.544, but that's misleading since it excludes the hardest-to-classify points.

K-Means was the most useful here, even with low silhouette scores, because the cluster profiles were interpretable (e.g., cluster of short weekday commutes vs longer weekend leisure trips). But honestly, the trip data is fairly continuous — there aren't hard boundaries between trip types.

## PCA

After one-hot encoding I had about 20 features, some correlated. PCA reduces this to uncorrelated components.

![PCA Explained Variance](assets/pca_variance.png)
*Two components capture 49.2% of variance; three capture 66.2%.*

Based on the loadings:
- **PC1** (28% variance): dominated by start_lat and start_lng — a location component capturing where trips start.
- **PC2** (21% variance): driven by trip_duration_min and trip_distance_km — trip intensity.
- **PC3** (17% variance): loaded on hour and day_of_week — time patterns.

Projecting the data into PC1–PC2 space shows overlap between members and casual riders, consistent with the 62% classification ceiling. PCA confirms what the classifiers told us: the signal is real but weak.

## Key Takeaways

The clearest result is that Random Forest was the only model that worked for both tasks. Linear models couldn't handle the non-linear relationships, and regression in particular punished them with negative R² — they were actively worse than guessing the average. That's not because linear regression is a bad algorithm, it's because trip duration doesn't scale linearly with any available feature. A 5-minute trip and a 50-minute trip can have the same distance and start location.

Regression turned out to be much harder than classification. Even the best model (R² = 0.28) leaves most of the variance unexplained. Looking back, that makes sense — trip duration depends on things I don't have: weather, rider age, route elevation, how long someone spent at a red light. You'd need more data to do better.

The negative R² scores were the most educational part of the project. They make it really obvious when a model is a bad fit. If I'd only looked at RMSE, I might have thought the linear models were just "kinda bad." The negative R² says they're systematically broken for this problem.

Clustering was the least satisfying section. Silhouette scores below 0.2 across all three methods — the trips just don't form clean groups. That's not a bug in the algorithms, it's an honest property of the data: bikeshare trips in a summer month are a smooth mix of commuting, errands, and leisure with no hard edges between them.

The 62% classification ceiling bugged me until I remembered it's July. In winter, casual riders basically disappear — the split would be more like 80/20 and the classification signal would be stronger. But July is peak tourism, so members and casuals both take short recreational trips along the lakefront. They're harder to tell apart. If I were doing this again, I'd compare summer and winter months side by side.

## How to Run

1. Install Python 3.9+ with pandas, numpy, scikit-learn, matplotlib, seaborn, pyarrow, scipy
2. Place `202307-divvy-tripdata.parquet` in the repo root
3. Open `bikeshare_notebook.ipynb` in Jupyter
4. Run all cells top to bottom

## Stack

Python, pandas, numpy, scikit-learn, matplotlib, seaborn, pyarrow, scipy
