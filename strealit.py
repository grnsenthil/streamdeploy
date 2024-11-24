import streamlit as st
import pandas as pd
import numpy as np
import joblib


#read the dataset to fill the values of input option of each element
df = pd.read_csv('train_LZdllcl.csv')

department = st.selectbox("Department", pd.unique(df['department']))
region = st.selectbox("Region", pd.unique(df['region']))
education = st.selectbox("Education", pd.unique(df['education']))
gender = st.selectbox("Gender", pd.unique(df['gender']))
recruitment_channel = st.selectbox("Recruitment_channel", pd.unique(df['recruitment_channel']))

no_of_trainings = st.number_input('no_of_trainings')
age = st.number_input('age')
previous_year_rating = st.number_input('previous_year_rating')
length_of_service = st.number_input('length_of_service')
KPIs_met = st.number_input('KPIs_met >80%')
awards_won = st.number_input('awards_won?')
avg_training_score = st.number_input('avg_training_score')


input = {
    'department':department,
    'region':region,
    'education':education,
    'gender':gender,
    'recruitment_channel':recruitment_channel,
    'no_of_trainings':no_of_trainings,
    'age':age,
    'previous_year_rating':previous_year_rating,
    'length_of_service':length_of_service,
    'KPIs_met >80%':KPIs_met,
    'awards_won?':awards_won,
    'avg_training_score':avg_training_score
}


model = joblib.load('model_pipeline_train_test')

if st.button('Predict'):
    X_inputs = pd.DataFrame(input,index=[0])
    prediction = model.predict(X_inputs)
    st.write('The predicted the value is:')
    st.write(prediction)

    # Post this we have to run the windows cmd prompt "streamlit run D:\DS\DataScience_Working\Deploy\Steamlit\strealit.py"