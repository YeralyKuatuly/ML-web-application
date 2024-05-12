import model_forest as mforest
import model_knn as mknn
import clustering as cluster
import streamlit as st
import pandas as pd
import os
from streamlit_pandas_profiling import st_profile_report
from pandas_profiling import ProfileReport



with st.sidebar:
    st.image("https://cdn-icons-png.freepik.com/512/8618/8618875.png")
    st.title("Predict Car Price")
    choice = st.radio("Navigation", ["Upload", "ML", "Profiling", "Download"])
    st.info("This Streamlit app will allow you to predict with your car parameters it's Price. Also you are able to "
            "choose model RandomForest, KNNeighbours for supervised learning and XXX for unsupervised learning.")

if os.path.exists("sourcedata.csv"):
    df = pd.read_csv("sourcedata.csv", index_col=None)

if choice == "Upload":
    st.title("Upload Your Data for Modelling!")
    file = st.file_uploader("Upload Your Dataset Here")
    if file is not None:
        df = pd.read_csv(file)
        df.to_csv("sourcedata.csv", index=False)
        st.dataframe(df)

if choice == "Profiling":
    st.title("Automated Exploratory Data Analysis")
    profile_report = df.profile_report()
    st_profile_report(profile_report)

if choice == "ML":
    st.title("Machine Learning Models")
    choice2 = st.selectbox("Select Model", ["RandomForest", "KNNeighbours", "KMeansClustering"])
    if choice2 == "RandomForest":
        mforest.head_text()
        mforest.input_parameters()
        mforest.curve_show()
    elif choice2 == "KNNeighbours":
        mknn.head_text()
        mknn.input_parameters()
        mknn.curve_show()
    else:
        cluster.head_text()
        cluster.plot1()
        cluster.do_cluster1()



if choice == "Download":
    # choice2 = st.selectbox("Select Model", ["RandomForest", "KNNeighbours"])
    st.title("There You can download used Machine Learning models")
    with open("knn_model.pkl", "rb") as f:
        st.download_button("Download Model", f, key="knn_model")

    st.subheader("Be free to share your opinion about this project.")
    user_feedback = st.text_area("Enter Your Feedback")
    if user_feedback:
        st.write("Thanks for your feedback!")
    print(user_feedback)
