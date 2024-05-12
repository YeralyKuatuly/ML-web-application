from sklearn.cluster import KMeans
import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt


data = pd.read_csv('csv_files/clean_data.csv')

def head_text():
    st.title('Unsupervised Learning - K-Means Clustering')
    st.image("https://cdn.hackr.io/uploads/posts/large/1600253014vJgLQIJ7nI.png")


print(data.head())

def plot1():
    fig, ax = plt.subplots()

    ax.scatter(data.Year, data.Price)
    ax.set_xlabel("Year")
    ax.set_ylabel("Price")
    ax.set_title("Car Year vs Price")

    st.pyplot(fig)

def do_cluster1():
    km = KMeans(n_clusters=5)
    y_predicted = km.fit_predict(data[['Year', 'Price']])
    print(y_predicted)

    data["cluster1"] = y_predicted

    df1 = data[data["cluster1"] == 0]
    df2 = data[data["cluster1"] == 1]
    df3 = data[data["cluster1"] == 2]
    df4 = data[data["cluster1"] == 3]
    df5 = data[data["cluster1"] == 4]

    fig, ax = plt.subplots()

    ax.scatter(df1.Year, df1['Price'], color='green', label='Cluster 1')
    ax.scatter(df2.Year, df2['Price'], color='red', label='Cluster 2')
    ax.scatter(df3.Year, df3['Price'], color='purple', label='Cluster 3')
    ax.scatter(df4.Year, df4['Price'], color='gray', label='Cluster 4')
    ax.scatter(df5.Year, df5['Price'], color='yellow', label='Cluster 5')
    ax.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], color='black', marker='*', label='Centroid')

    # Set labels and legend
    ax.set_xlabel('Year')
    ax.set_ylabel('Price')
    ax.legend()

    # Display the plot using Streamlit
    st.pyplot(fig)



