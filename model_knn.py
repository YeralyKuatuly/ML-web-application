import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import streamlit as st
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


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
    st.title("K-Nearest Neighbours")
    st.image("https://res.cloudinary.com/dtoupvd2u/image/upload/v1683192969/knn_algorithm_l_a1fb23f3ee.jpg")


def predicting(data):

    param_grid = {'algorithm': ['brute'], 'metric': ['euclidean'], 'n_neighbors': [5]}
    knn = KNeighborsRegressor()

    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='r2')
    grid_search.fit(X_train, y_train)

    # Get the best parameters
    best_params = grid_search.best_params_

    # Train the model with the best parameters
    knn_best = grid_search.best_estimator_
    # Instantiate the grid search model
    y_pred = knn_best.predict(X_test)
    r2_test = r2_score(y_test, y_pred)
    st.write(f'R-squared Score on Test Data: {r2_test * 100:.2f}')

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
    st.image("pages/curve_knn.png")
