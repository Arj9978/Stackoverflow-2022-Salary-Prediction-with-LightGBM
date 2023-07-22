import streamlit as st
import numpy as np
import pandas as pd
import joblib
import lightgbm

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,LabelEncoder, OneHotEncoder

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
    X_new = np.array([country,education,expericence])
    X_new_df = pd.DataFrame([X_new], columns = columns)

    categorical_pipeline = Pipeline([('OneHot', OneHotEncoder(handle_unknown='ignore'))])

    numeric_pipeline = Pipeline([('Scaler', StandardScaler())])  
                                 
    transformer = ColumnTransformer([('Num', numeric_pipeline, ['YearsCodePro']), 
                                 ('Category', categorical_pipeline, ['Country','EdLevel'])])
    
    X_new_df = transformer.fit_transform(X_new_df)
    st.write(X_new_df.shape)
    salary = model.predict(X_new_df)
    
    st.subheader(f"The estimated salary is ${salary[0]:.2f}")\

