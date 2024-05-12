import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import joblib
import streamlit as st

# Load the data
df = pd.read_csv('csv_files/clean_data.csv')

# Function to preprocess the data
def preprocess_data(df):
    # Separating out the features and target variable
    X = df.drop('Price', axis=1)  # Assuming 'Price' is the target variable
    y = df['Price']

    # Standardizing the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Applying PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)

    return X_pca, y

# Preprocess the data
X_pca, y = preprocess_data(df)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Function to train KNeighborsRegressor
def train_knn(X_train, y_train):
    # Initialize KNeighborsRegressor
    knn = KNeighborsRegressor()

    # Hyperparameter grid for GridSearchCV
    param_grid = {
        'n_neighbors': range(1, 50),
        'metric': ['euclidean', 'manhattan', 'minkowski'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
    }

    # Perform grid search for hyperparameter tuning
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='r2')
    grid_search.fit(X_train, y_train)

    # Get the best parameters
    best_params = grid_search.best_params_

    # Train the model with the best parameters
    knn_best = grid_search.best_estimator_

    return knn_best, best_params

# Train KNeighborsRegressor
knn_best, best_knn_params = train_knn(X_train, y_train)

# Function to train RandomForestRegressor
def train_random_forest(X_train, y_train):
    # Define the parameter grid to search
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 4, 6],
        'min_samples_leaf': [1, 2, 3],
        'max_features': ['sqrt', 'log2']  # Use valid options for max_features
    }

    # Create a base model
    rf = RandomForestRegressor()

    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                               cv=3, n_jobs=-1, verbose=2)

    # Fit the grid search to the data
    grid_search.fit(X_train, y_train)

    # Best parameters
    best_rf_params = grid_search.best_params_

    # Train the model with the best parameters
    rf_best = grid_search.best_estimator_

    return rf_best, best_rf_params

# Train RandomForestRegressor
rf_best, best_rf_params = train_random_forest(X_train, y_train)

# Function to save trained model
def save_model(model, filename):
    joblib.dump(model, filename)

# Save the trained models
save_model(knn_best, "knn_model.pkl")
save_model(rf_best, "random_forest_model.pkl")

# Function to load saved model
def load_model(filename):
    return joblib.load(filename)

# Load the saved models
loaded_knn_model = load_model("knn_model.pkl")
loaded_rf_model = load_model("random_forest_model.pkl")

# Streamlit app
st.title("Car Price Prediction")

# Sidebar for user input
st.sidebar.header("Choose Model")
model_choice = st.sidebar.selectbox("Select Model", ["KNN", "Random Forest"])

if model_choice == "KNN":
    model = loaded_knn_model
    params = best_knn_params
else:
    model = loaded_rf_model
    params = best_rf_params

st.sidebar.subheader("Model Parameters")
st.sidebar.write(params)

# Predictions
st.subheader("Predictions")
predicted_prices = model.predict(X_test)
st.write("Predicted Prices:", predicted_prices)

# Evaluation
mse = mean_squared_error(y_test, predicted_prices)
r2 = r2_score(y_test, predicted_prices)
st.subheader("Model Evaluation")
st.write("Mean Squared Error:", mse)
st.write("R-squared Score:", r2)

# Learning Curve
train_sizes, train_scores, val_scores = learning_curve(model, X_train, y_train, cv=3, scoring='r2')
train_scores_mean = np.mean(train_scores, axis=1)
val_scores_mean = np.mean(val_scores, axis=1)

# Plot Learning Curve
st.subheader("Learning Curve")
plt.figure()
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, val_scores_mean, 'o-', color="g", label="Cross-validation score")
plt.xlabel("Training examples")
plt.ylabel("R-squared score")
plt.title("Learning Curves")
plt.legend(loc="best")
plt.grid()
st.pyplot(plt)
