import streamlit as st
import numpy as np
import pandas as pd
import pickle
#import joblib


# st.cache_resource
# def load_model():
#     with open('Titanic_model.pkl', 'rb') as f:
#         return pickle.load(f)
    
# Titanic_model = load_model()


import pandas as pd
#import seaborn as sns
import numpy as np

import warnings

warnings.filterwarnings('ignore')

titanic = pd.read_csv('titanic.csv')

# titanic

# titanic.info()

#### Data cleaning

titanic['age'].mean()
titanic['age'].fillna(titanic['age'].mean())
titanic['age'] = titanic['age'].fillna(titanic['age'].mean())

# titanic.info()

titanic['embarked'].isna()
index = titanic[titanic['embarked'].isna()].index

titanic.drop(index,inplace=True)

# titanic.info()

titanic.drop('deck',axis=1,inplace=True)

# titanic.info()

#### Features Selection

# sns.pairplot(titanic)

y = titanic['survived']

x = titanic.drop(['survived', 'who', 'adult_male', 'alive', 'alone'],axis=1)


#### Splitting data into training and testing

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=42)

# x_train

#### Feature engineering

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

label = LabelEncoder()

print(titanic['class'])

x['class_label'] = label.fit_transform(titanic['class'])

# x.head()

onehot = OneHotEncoder()

onehot.fit_transform(titanic[[  'sex'  ]])

pd.get_dummies(titanic[['sex', 'embarked', 'embark_town']])

pd.get_dummies(x[['sex', 'embarked', 'embark_town']],dtype=int)

encoded = pd.get_dummies(x[['sex', 'embarked', 'embark_town']],dtype=int,drop_first=True)

# encoded

x = pd.concat([x,encoded],axis=1)

# x.head()

x.drop(  ['sex', 'class', 'embark_town', 'embarked'],  axis=1,inplace=True)

# x.head()

#### Splitting into training and testing

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=42)

# x_train

#### Algorithm selection

from sklearn.linear_model import LogisticRegression

Titanic_model = LogisticRegression()

Titanic_model.fit(x_train, y_train)

predictions = Titanic_model.predict(x_test)

# predictions

table = pd.DataFrame(predictions,columns=['Predicitons'])
y_test.reset_index(drop=True)
table['y_test'] = y_test.reset_index(drop=True)
# table

### evaluation

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# confusion_matrix(y_test,predictions)

# sns.heatmap(confusion_matrix(y_test,predictions),annot=True)

print(classification_report(y_test, predictions))

accuracy_score(y_test,predictions)




st.title('Titanic Survival Prediction App')
st.write('This app will predict Titanic Accident Survival Rate')

st.subheader('Survived versus Sex, Age, Class, Fare, Embarked Town')

age = st.number_input('Age', 0, 100)
sex = st.selectbox('Gender', [0, 1], index=None, placeholder='Select gender')
Fare = st.number_input('Ship Fare', 70, 250)
Class = st.selectbox('Cabin Class', [1, 2, 3], index=None, placeholder='Select class')
Parch = st.selectbox('Parch Number', ['0', '1', '2'], index=None)
# Embarked_town = st.selectbox('Pickup Town', ['Southampton', 'Cherbourg', 'Queenstown'], index=None, placeholder='Choose pickup location')

input = np.array([age,sex,Fare,Class,Parch]).reshape(1,-1)

if st.button('Predict'):
    result = Titanic_model.predict(input)
    st.write(result)

#prediction = Titanic_model.predict(features)
#if prediction == 1:
    #st.success('Survived(1)')
#else:
    #st.error('Did not survive(0)')

