# Group 3 DL — Movie Recommendation Scoring Model

## Project Overview
This project builds an Artificial Neural Network (ANN) to predict how much a user will like a movie on a **1–5 rating scale**. The predicted scores can then be used to recommend the highest-rated unseen movies to each user.

## Problem Statement
Users cannot review every movie in a catalog. A recommendation model can learn patterns from previous user–movie ratings and estimate which unseen movies a user may enjoy.

## Dataset
**MovieLens 100K** (GroupLens): 100,000 ratings from **943 users** across **1,682 movies**. The notebook uses user IDs, movie IDs, ratings, and movie titles.

## Project Workflow
1. **Data collection and EDA:** Download MovieLens 100K; examine rating distribution, ratings per user/movie, and sparsity.
2. **Preprocessing:** Map user and movie IDs to consecutive integer indices for embedding layers.
3. **Data splitting:** Random **80% training / 10% validation / 10% testing** split (`random_state=42`).
4. **Baselines:** Compare against the global mean rating and each movie's mean training rating.
5. **Modeling:** Train an ANN using user and movie embeddings, concatenation, Dense layers with ReLU, and a single rating output.
6. **Regularization:** Use L2, Dropout, and Early Stopping to reduce overfitting.
7. **Hyperparameter experiments:** Compare the base ANN with individual configurations from five team members.
8. **Evaluation:** Measure test RMSE, MAE, and ROC-AUC for ratings of 4–5 treated as a positive preference.
9. **Model persistence and inference:** Save/reload the Keras model and ID mappings; predict a rating or recommend unseen movies.

## ANN Architecture
- Separate **user embedding** and **movie embedding** inputs
- Flatten and concatenate the embedding vectors
- Configurable Dense hidden layers with **ReLU** activation
- **Dropout** and **L2 regularization**
- One linear output for predicted rating
- **Adam** optimizer and regression loss

## Team Hyperparameter Experiments
Each experiment uses the same dataset and train/validation/test split, while changing model hyperparameters.

| Experiment | Embedding size | Hidden layers | Units | Dropout | L2 | Learning rate | Batch size |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base ANN | 32 | 2 | 64 | 0.2 | 0.00001 | 0.001 | 256 |
| Norah | 32 | 2 | 64 | 0.2 | 0.00001 | 0.0003 | 256 |
| Shaden | 64 | 2 | 128 | 0.4 | 0.00001 | 0.0003 | 512 |
| Raghad | 48 | 2 | 96 | 0.3 | 0.000003 | 0.0002 | 256 |
| Basma | 32 | 3 | 64 | 0.2 | 0.00001 | 0.0003 | 256 |
| Rawan | 64 | 3 | 128 | 0.4 | 0.00001 | 0.0005 | 128 |

## Evaluation and Comparison
- **RMSE (lower is better):** Penalizes larger errors in predicted movie ratings.
- **MAE (lower is better):** Average absolute difference between actual and predicted ratings.
- **ROC-AUC (higher is better):** Measures ranking quality for movies rated **4 or 5** versus lower ratings.

The notebook's final comparison evaluates **Movie Average, Base ANN, Norah, Shaden, Raghad, Basma, and Rawan**, and includes training epochs. The final Rawan-inclusive comparison is computed when the notebook is executed; no unverified results are reported here.

## Limitations
- **Cold start:** New users and movies do not have learned embeddings.
- The model uses IDs and ratings, not genres, movie descriptions, or demographic features.
- Ratings reflect which movies users chose to watch and rate, introducing selection bias.
- MovieLens 100K is a relatively small, older dataset; results may not transfer directly to modern platforms.
