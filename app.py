import streamlit as st
import numpy as np
import pandas as pd
import pickle
#import joblib


st.cache_resource
def load_model():
    with open('Titanic_model.pkl', 'rb') as f:
        return pickle.load(f)
    
Titanic_model = load_model()


st.title('Titanic Survival Prediction App')
st.write('This app will predict Titanic Accident Survival Rate')

st.subheader('Survived versus Sex, Age, Class, Fare, Embarked Town')

age = st.number_input('Age', 0, 100)
sex = st.selectbox('Gender', ['Male', 'Female'], index=None, placeholder='Select gender')
Fare = st.number_input('Ship Fare', 70, 250)
Class = st.selectbox('Cabin Class', ['1st', '2nd', '3rd'], index=None, placeholder='Select class')
Parch = st.selectbox('Parch Number', ['0', '1', '2'], index=None)
Embarked_town = st.selectbox('Pickup Town', ['Southampton', 'Cherbourg', 'Queenstown'], index=None, placeholder='Choose pickup location')

if st.button('Predict'):
    features = [['class', 'age', 'fare', 1 if 'sex' == 'Male' else 0]]

#prediction = Titanic_model.predict(features)
#if prediction == 1:
    #st.success('Survived(1)')
#else:
    #st.error('Did not survive(0)')

