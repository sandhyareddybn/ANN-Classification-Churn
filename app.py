import pickle
import tensorflow as tf
import streamlit as st
import pandas as pd

with open('gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('geography.pkl', 'rb') as file:
    onehotencoded_geography = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

loaded_model = tf.keras.models.load_model('churn_model_v2.keras')

st.title('Customer Churn Prediction')

#user input
credit_score = st.number_input('Credit Score')
geography = st.selectbox('Geography', onehotencoded_geography.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age')
tenure = st.slider('Tenure')
balance = st.number_input('Balance')
num_of_products = st.number_input('Number of Products')
has_credit_card = st.selectbox('Has Credit Card', [1, 0])
is_active_member = st.selectbox('Is Active Member', [1, 0])
estimated_salary = st.number_input('Estimated Salary')

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Geography': [geography],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_credit_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]

})

geo_encoded = onehotencoded_geography.transform(input_data[['Geography']]).toarray()
geo_names = onehotencoded_geography.get_feature_names_out(['Geography'])
geo_df = pd.DataFrame(geo_encoded, columns=geo_names)

input_data = pd.concat([input_data.drop('Geography', axis=1), geo_df], axis=1)

input_data_scaled = scaler.transform(input_data)

loaded_model.predict(input_data_scaled)

#probability of churn
churn_probability = loaded_model.predict(input_data_scaled)[0][0]

print(f'Churn Probability: {churn_probability}')

if churn_probability > 0.5:
    st.write(f'Churn Probability: {churn_probability:.2f} - Likely to Churn')
else:
    st.write(f'Churn Probability: {churn_probability:.2f} - Unlikely to Churn')


