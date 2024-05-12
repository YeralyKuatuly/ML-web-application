import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import streamlit as st
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

#Best parameters for forest: {'max_depth': 20, 'max_features': 'log2', 'min_samples_leaf': 1,
# 'min_samples_split': 2, 'n_estimators': 200}

data = pd.read_csv('csv_files/clean_data.csv')

X = data.drop('Price', axis=1)  # Assuming 'Price' is the target variable
y = data['Price']

# Standardizing the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA()
X_pca = pca.fit_transform(X_scaled)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


def head_text():
    st.title("Random Forest Regressor")
    st.image("https://www.analytixlabs.co.in/blog/wp-content/uploads/2023/09/Random-Forest-Regression.jpg")


def predicting(data):
    param_grid = {'max_depth': [20], 'max_features': ['log2'], 'min_samples_leaf': [1],
                  'min_samples_split': [2], 'n_estimators': [200]}
    rf = RandomForestRegressor()

    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    best_rf = grid_search.best_params_
    y_pred = grid_search.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    st.write(f'R-squared Score on Test Data: {r2 * 100:.2f}')
    data['Name'] = le.fit_transform(data['Name'])
    data['Body'] = le.fit_transform(data['Body'])
    data['Transmission'] = le.fit_transform(data['Transmission'])
    data['Customs Clearance'] = le.fit_transform(data['Customs Clearance'])
    st.success(f'Approximately your car will cost {int(grid_search.predict(data))} tenge')


def input_parameters():
    car_models = []
    with open("car_models.txt", "r") as file:
        for line in file:
            for element in line.strip().split(","):
                car_models.append(element)

    selected_car = st.selectbox("Select Car Model", options=car_models, index=0)

    body = st.selectbox("Body", ['кроссовер', 'седан', 'лифтбек', 'хетчбэк', 'универсал', 'внедорожник',
                                 'минивэн', 'пикап', 'лимузин', 'купе', 'кабриолет', 'фургон', 'микроавтобус'])
    year = st.number_input("Year", min_value=1900, max_value=2024, step=1)
    transmission = st.selectbox("Transmission", ['Автомат', 'Робот', 'Механика', 'Вариатор'])
    mileage = st.number_input("Mileage", min_value=0, step=1000)
    volume = st.slider("Volume", 0.0, 10.0, step=0.1)
    customs_clearance = st.checkbox("Customs Clearance", ["Yes", "No"])
    color = st.text_input("Color")
    drive = st.selectbox("Drive", ["Rear", "Front", "All-wheel"])

    data = {
        "Name": [selected_car],
        "Body": [body],
        "Year": [year],
        "Transmission": [transmission],
        "Mileage": [mileage],
        "Volume": [volume],
        "Customs Clearance": [customs_clearance],
        "Color": [1],
        "задний": [1],
        "передний": [0],
        "полный": [0]
    }

    df = pd.DataFrame(data)

    predicting(df)


def curve_show():
    st.image("pages/curve_forest.png")
