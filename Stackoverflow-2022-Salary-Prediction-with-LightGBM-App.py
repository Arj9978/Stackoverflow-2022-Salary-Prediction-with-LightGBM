import streamlit as st
import numpy as np
import pandas as pd
import joblib
import lightgbm
from utils import PrepProcesor, columns

model = joblib.load('finalized_model.joblib')
st.title('Salary Prediction in 2022')
st.write("""### We need some information to predict the salary""")

countries = (
    "United States of America",
    "Iran, Islamic Republic of...",
    "India",
    "United Kingdom of Great Britain and Northern Ireland",
    "Germany",
    "Canada",
    "Brazil",
    "France",
    "Spain",
    "Australia",
    "Netherlands",
    "Poland",
    "Italy",
    "Russian Federation",
    "Sweden",
    "Switzerland",
    "Israel",
    "Austria",
    "Portugal",
    "Denmark",
    "Turkey",
    "Belgium",
    "Norway",
    "Finland",
    "Greece",
    "Czech Republic",
    "New Zealand",
    "Mexico",
    "South Africa",
    "Pakistan",
    "Other"
)

education = (
    "Less than a Bachelor",
    "Bachelor’s degree",
    "Master’s degree"
)

country = st.selectbox("Country", countries)
education = st.selectbox("Education Level", education)
expericence = st.slider("Years of Experience", 0, 50, 3)

columns = ['Country', 'EdLevel', 'YearsCodePro']

ok = st.button("Calculate Salary")
if ok: 
    # Create a DataFrame with the row data and columns matching the training data
    expericence_input = pd.DataFrame(expericence, columns = ['YearsCodePro'])

    # Perform one-hot encoding for the countries and education input
    country_dummy = pd.get_dummies([country], columns=['Country'], prefix='', prefix_sep='')
    education_dummy = pd.get_dummies([education], columns=['EdLevel'], prefix='', prefix_sep='')

    # Align the countries and education columns with the training data columns
    country_dummy = country_dummy.reindex(columns=Country, fill_value=0)
    education_dummy = education_dummy.reindex(columns=EdLevel, fill_value=0)

    # Concatenate the address columns with the input data
    X = pd.concat([country_dummy, education_dummy, expericence_input], axis=1)
    st.write(X)

    X = np.array(X)
    st.write(X)


    
    
    # st.write(X_new_df.shape)
    # X_new_df = transformer.fit_transform(X_new_df)
    # st.write(X_new_df.shape)
    
    # salary = model.predict(X_new_df)
    
    st.subheader(f"The estimated salary is ${salary[0]:.2f}")
