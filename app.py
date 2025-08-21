import streamlit as st
import tensorflow as tf
import numpy as np
import joblib
import pandas as pd 

try:
    model = tf.keras.models.load_model('model.keras')
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()


try:
    scaler = joblib.load('scaler.pkl')
except Exception as e:
    st.error(f"Error loading the scaler: {e}")
    st.stop()

st.title('Water Potability Prediction')

st.write("""Enter the water quality parameters to predict if the water is potable.""")

default_ph_min, default_ph_max, default_ph_value = 0.0, 14.0, 7.0
default_hardness_min, default_hardness_max, default_hardness_value = 0.0, 400.0, 200.0
default_solids_min, default_solids_max, default_solids_value = 0.0, 70000.0, 20000.0
default_chloramines_min, default_chloramines_max, default_chloramines_value = 0.0, 20.0, 7.0
default_sulfate_min, default_sulfate_max, default_sulfate_value = 0.0, 500.0, 300.0
default_conductivity_min, default_conductivity_max, default_conductivity_value = 0.0, 1000.0, 400.0
default_organic_carbon_min, default_organic_carbon_max, default_organic_carbon_value = 0.0, 30.0, 14.0
default_trihalomethanes_min, default_trihalomethanes_max, default_trihalomethanes_value = 0.0, 140.0, 60.0
default_turbidity_min, default_turbidity_max, default_turbidity_value = 0.0, 10.0, 4.0


ph = st.slider('pH', min_value=default_ph_min, max_value=default_ph_max, value=default_ph_value)
hardness = st.slider('Hardness', min_value=default_hardness_min, max_value=default_hardness_max, value=default_hardness_value)
solids = st.slider('Solids', min_value=default_solids_min, max_value=default_solids_max, value=default_solids_value)
chloramines = st.slider('Chloramines', min_value=default_chloramines_min, max_value=default_chloramines_max, value=default_chloramines_value)
sulfate = st.slider('Sulfate', min_value=default_sulfate_min, max_value=default_sulfate_max, value=default_sulfate_value)
conductivity = st.slider('Conductivity', min_value=default_conductivity_min, max_value=default_conductivity_max, value=default_conductivity_value)
organic_carbon = st.slider('Organic Carbon', min_value=default_organic_carbon_min, max_value=default_organic_carbon_max, value=default_organic_carbon_value)
trihalomethanes = st.slider('Trihalomethanes', min_value=default_trihalomethanes_min, max_value=default_trihalomethanes_max, value=default_trihalomethanes_value)
turbidity = st.slider('Turbidity', min_value=default_turbidity_min, max_value=default_turbidity_max, value=default_turbidity_value)


if st.button('Predict Potability'):
    input_data = pd.DataFrame([[ph, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, trihalomethanes, turbidity]],
                               columns=['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity'])


    scaled_input_data = scaler.transform(input_data)
    model_input_shape = model.input_shape
    if len(model_input_shape) == 3: 
         reshaped_input_data = scaled_input_data.reshape(scaled_input_data.shape[0], scaled_input_data.shape[1], 1)
    else: 
         reshaped_input_data = scaled_input_data

    prediction = model.predict(reshaped_input_data)
    prediction_proba = prediction[0][0]

    if prediction_proba > 0.5:
        st.success(f'The water is predicted to be Potable (Probability: {prediction_proba:.4f})')
    else:
        st.error(f'The water is predicted to be Not Potable (Probability: {prediction_proba:.4f})')