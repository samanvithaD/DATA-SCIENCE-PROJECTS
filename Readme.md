# Data Science Projects

Three end-to-end machine learning projects in Python, each done in a Jupyter notebook (data cleaning, EDA, modelling, evaluation, business recommendations), plus a Streamlit app that demos them.

| Project | Problem | Tools |
|---|---|---|
| [Customer Attrition Prediction](#1-customer-attrition-prediction) | Classification | Pandas, Scikit-learn, XGBoost, Matplotlib, Seaborn |
| [Time Series Forecasting of Retail Sales](#2-time-series-forecasting-of-retail-sales) | Forecasting | Statsmodels, Prophet, XGBoost, Pandas |
| [Movie Recommendation System](#3-movie-recommendation-system) | Content-based recommendation | Scikit-learn (TF-IDF), Pandas, WordCloud |

**Live demo:** _add your Streamlit link here_

---

## 1. Customer Attrition Prediction

Notebook: `notebooks/Customer_Attrition_Prediction.ipynb`

**Goal:** predict which telecom customers will leave, find the drivers, and recommend retention actions.

**Data:** 3,333 customers, 11 columns, 14.5% churn. No missing values or duplicates.

**What I did**
1. Loaded and checked the data (missing values, duplicates, summary statistics).
2. EDA: churn rate by contract renewal, customer-service calls and daytime usage, plus a correlation heatmap.
3. Feature engineering: average call length, overage ratio, charge per day minute, and a flag for 4+ service calls.
4. Stratified 80/20 train-test split to keep the 14.5% churn rate in both sets.
5. Trained Logistic Regression, Random Forest and XGBoost, handling class imbalance with class weights.
6. Evaluated with accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrices and ROC curves.
7. Ranked churn drivers with odds ratios, feature importance and permutation importance.
8. Scored every customer with 5-fold cross-validated churn probabilities and grouped them into Low, Medium and High risk tiers.
9. Identified high-churn segments and wrote recommendations and limitations.

**Results**

| Model | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.825 | 0.588 | 0.851 | 0.454 |
| Random Forest | 0.670 | 0.670 | 0.863 | 0.720 |
| XGBoost | 0.742 | 0.646 | 0.866 | 0.739 |

- Customers with 4+ service calls churn at **51.7%** (baseline group: 8.2%).
- Customers who did not renew churn at **42.4%**.
- The High-risk tier churned at about 72%, against about 3% for Low.

**Recommendations:** escalate customers at the 3rd service call, offer renewal incentives, review plans for heavy daytime users, and contact the High-risk tier first.

---

## 2. Time Series Forecasting of Retail Sales

Notebook: `notebooks/Time_Series_Forecasting_of_Retail_sales.ipynb`

**Goal:** forecast daily demand to support stock planning.

**Data:** 73,100 rows covering 731 days (2022-01-01 to 2024-01-01), 5 stores and 20 products.

**What I did**
1. Cleaned and checked the data (missing values, duplicates, negative units, missing dates, outliers).
2. Aggregated sales into a daily series and added time features.
3. Stationarity testing (ADF), rolling statistics, seasonal decomposition, and ACF/PACF analysis.
4. ANOVA tests for weekday, month and season effects, and checks on promotion, price and weather.
5. Chronological 80/20 train-test split.
6. Built simple baselines (last value, same day last week, historical mean).
7. Modelled with ARIMA, SARIMA, Prophet (with and without promotion, price and weather factors) and XGBoost on calendar features.
8. Compared models with MAE, RMSE and MAPE, then checked accuracy at weekly and monthly level and for categories, stores and products.
9. Produced a 90-day forecast with prediction intervals and a safety stock estimate.

**Results**

| Model | MAPE % |
|---|---|
| SARIMA / ARIMA / Historical mean | 6.27 |
| Prophet | 6.26 |
| XGBoost | 6.32 |
| Prophet + external factors | 6.38 |

- The series is stationary, and weekday, month and season effects are not significant.
- No model clearly beat the historical mean, so daily sales behave like noise around a stable level.
- Aggregating helps: weekly MAPE **1.95%** and monthly **1.07%**. Error grows for smaller groups such as individual products.

**Recommendation:** plan weekly or monthly, with safety stock based on forecast error.

---

## 3. Movie Recommendation System

Notebook: `notebooks/Movie_Recommendation_System.ipynb`

**Goal:** recommend movies similar to one the user likes, using content only.

**Data:** three datasets (movie metadata, credits, keywords) merged and filtered to 9,141 movies with at least 50 votes.

**What I did**
1. Cleaned and merged the datasets (fixed bad ids, removed duplicates).
2. Parsed nested "stringified list" columns with `ast.literal_eval` to extract cast, director, genres and keywords.
3. Normalised the text (lowercase, names joined into one token) and built a combined `tags` column from the top 3 cast, director, genres and keywords.
4. EDA: genre and actor charts and a keyword word cloud.
5. Compared CountVectorizer and TF-IDF with cosine similarity, using proxy metrics against a random baseline.
6. Built a `recommend` function returning the 10 most similar movies, with case-insensitive search and "did you mean" suggestions for misspelt titles.
7. Plotted a similarity heatmap for a small set of movies.

**Results:** TF-IDF found movies sharing a cast member or director far more often than CountVectorizer (0.49 against 0.19; random 0.003), so it was chosen. For example, *Toy Story* returns its sequels and related shorts.

---

## Streamlit demo app

The app is a simplified, interactive version of the notebooks. The full analysis stays in the notebooks.

| Page | What it shows |
|---|---|
| Customer Attrition | Model comparison, churn driver charts, and a form that scores a customer |
| Retail Forecasting | Baselines vs ARIMA, forecast with prediction interval, safety stock sliders |
| Movie Recommender | Search box with typo suggestions |

Notebook-only work: odds ratios, permutation importance, SARIMA, Prophet, XGBoost forecasting, and the similarity heatmap.

**Run locally**

```bash
python -m pip install -r requirements.txt
python -m streamlit run Home.py
```

## Repository structure

```
├── Home.py
├── pages/                  # Streamlit pages
├── data/                   # telecom_churn.csv, retail_store_inventory.csv, movies_slim.csv
├── tools/make_slim_data.py # rebuilds movies_slim.csv from the full movie CSVs
├── notebooks/              # the three analysis notebooks
└── requirements.txt
```

`movies_slim.csv` holds the cleaned tags for the 9,141 movies, created with the same steps as the notebook.

## Limitations and next steps

- **Churn:** hyperparameters were set by hand and the test set has only 97 churners, so differences between models are small. Next: cross-validated tuning and a calibration check.
- **Forecasting:** the data shows no trend or seasonality, so results may not carry over to a real store. Next: walk-forward validation and lag features.
- **Recommender:** content-based only, so every user gets the same list. Next: use the plot overview text and add collaborative filtering.

## Author

**Dharmavarapu Sita Samanvitha**, B.Tech in Computer Science Engineering, KL University
Email: samanvitha1228@gmail.com | LinkedIn: _add your profile link_