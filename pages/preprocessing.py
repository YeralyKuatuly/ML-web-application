import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()  #change values to numbers


st.title("Data Preprocessing")
data = pd.read_csv("car_info_file.csv")
st.subheader("Car Data")
st.dataframe(data)
st.subheader("Data Info")

info_describe = data.describe()
info_shape = data.shape

# Display the output using st.code() within an expander
with st.expander("Data Describe"):
    st.code(info_describe)

with st.expander("Data Shape"):
    st.code(info_shape)

st.subheader("Some cleaning")
with st.echo():
    #remove rows that has Null values
    data_cleaned = data[data.apply(lambda row: 'Null' not in row.values, axis=1)]  
    #not necessary column
    data_cleaned = data_cleaned.join(pd.get_dummies(data_cleaned['Car Drive'])).drop(['Car Drive'], axis=1)  
    #make it numeric
    data_cleaned['Customs Clearance'] = data_cleaned['Customs Clearance'].apply(lambda x: 1 if x == 'Да' else 0)  

st.dataframe(data_cleaned.iloc[:3])

st.subheader("from sklearn.preprocessing import LabelEncoder")
st.info("It makes values more suitable for machine learning")


with st.echo():
    data_cleaned['Steering wheel'] = le.fit_transform(data_cleaned['Steering wheel'])
    data_cleaned['Transmission'] = le.fit_transform(data_cleaned['Transmission'])
    data_cleaned['Body'] = le.fit_transform(data_cleaned['Body'])
    data_cleaned['Color'] = le.fit_transform(data_cleaned['Color'])
    data_cleaned['Name'] = le.fit_transform(data_cleaned['Name'])

st.dataframe(data_cleaned.iloc[:3])
info_shape = data_cleaned.shape
st.latex("data\_cleaned.shape()")
st.code(info_shape)

st.subheader("Changes with incorrect collected data")
with st.echo():
    #  Remove string and make it integer
    data_cleaned['Mileage'] = data_cleaned['Mileage'].str.extract('(\d+)').astype('float')
    #  Remove string and make it integer
    data_cleaned['Volume'] = data_cleaned['Volume'].str[:-2].astype('float')
    #  Drop not necessary column
    data_cleaned = data_cleaned.drop(['Steering wheel'], axis = 1)


st.subheader("Remove Null values")
st.code("data_cleaned = data_cleaned[data_cleaned.apply(lambda row: 'Null' not in row.values, axis=1)]")

st.subheader("Remove Outliers")
with st.echo():
    iqr_multiplier = 1.5

    Q1 = data_cleaned[['Price']].quantile(0.25)
    Q3 = data_cleaned[['Price']].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - iqr_multiplier * IQR
    upper_bound = Q3 + iqr_multiplier * IQR

    outliers_mask = ((data_cleaned[['Price']] < lower_bound) |
                     (data_cleaned[['Price']] > upper_bound)).any(axis=1)

    data_cleaned = data_cleaned[~outliers_mask]

info_shape = data_cleaned.shape
st.latex("data\_cleaned.shape()")
st.code(info_shape)

st.dataframe(data_cleaned)
