import pickle
import streamlit as st
import pandas as pd
import tensorflow as tf

with open('gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('geography.pkl', 'rb') as file:
    onehotencoded_geography = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

loaded_model = tf.keras.models.load_model('churn_model_v2.keras')

st.title('Customer Churn Prediction')

#user input
CreditScore = st.number_input('Credit Score')
Geography = st.selectbox('Geography', onehotencoded_geography.categories_[0])
Gender = st.selectbox('Gender',label_encoder_gender.classes_)
Age = st.slider('Age')
Tenure = st.slider('Tenure')
Balance = st.number_input('Balance')
NumOfProducts = st.number_input('Number of Products')
HasCreditCard = st.selectbox('Has Credit Card', [1,0])
IsActiveMember = st.selectbox('Is Active Member', [1,0])
EstimatedSalary = st.number_input('Estimated Salary')

input_data = pd.DataFrame({
    'CreditScore': [CreditScore],
    'Geography': [Geography],
    'Gender': [label_encoder_gender.transform([Gender])][0],
    'Age': [Age],
    'Tenure': [Tenure],
    'Balance': [Balance],
    'NumOfProducts': [NumOfProducts],
    'HasCrCard': [HasCreditCard],
    'IsActiveMember': [IsActiveMember],
    'EstimatedSalary': [EstimatedSalary]
})

geo_encoded = onehotencoded_geography.transform(input_data[['Geography']]).toarray()
geo_names = onehotencoded_geography.get_feature_names_out(['Geography'])
geo_df = pd.DataFrame(geo_encoded, columns=geo_names)

input_data_df = pd.concat([input_data.drop('Geography', axis=1), geo_df], axis=1)

input_data_scaled = scaler.transform(input_data_df)

prediction_probability = loaded_model.predict(input_data_scaled)[0][0]

if prediction_probability > 0.5:
    st.write(f'Churn probabilty -{prediction_probability:.2f}- Customer is likely to churn.')
    print(f'Churn Probability: {prediction_probability:.2f}- Customer is likely to churn.')
else:
    st.write(f'Churn probabilty -{prediction_probability:.2f}- Customer is not likely to churn.')
    print(f'Churn Probability: {prediction_probability:.2f}- Customer is not likely to churn.')
