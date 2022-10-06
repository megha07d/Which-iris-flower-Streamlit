
import streamlit as st #do st things
import pandas as pd #process dataframes

from sklearn import datasets #get toy datsets

from sklearn.ensemble import RandomForestClassifier


st.write("""
# Simple iris prediction app

This app predicts the **Iris flower ** type!
""")

st.sidebar.header('User input Parameters')

def user_input_features():
    # sntax : start , end , default
    sepal_len = st.sidebar.slider('Sepal length',4.3,7.9,5.4)
    sepal_wid = st.sidebar.slider('Sepal width',2.0,4.4,3.4)
    petal_len = st.sidebar.slider('Petal Length',1.0,6.9,1.3)
    petal_wid = st.sidebar.slider('Ptal wodth',0.1,2.5,0.2)

    data = {'sepal_length':sepal_len,'sepal_width':sepal_wid,'petal_length':petal_len,'petal_width':petal_wid}

    features = pd.DataFrame(data,index=[0])

    return features

df = user_input_features()

st.subheader('User input parameters')
st.write(df)

iris = datasets.load_iris() #directly get training data from datasets module

X = iris.data
Y = iris.target

clf = RandomForestClassifier()  #here re 4 classes
clf.fit(X,Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.write('Class labels and corresponding index numbers')
st.write(iris.target_names)

st.subheader('Prediction')
# st.write(prediction)    #gives inx-coz one hot encoded?
st.write(iris.target_names[prediction])

st.subheader('Prection probability')
st.write(prediction_proba)

#cloned to local sys