import streamlit as st
import pickle
import numpy as np

# Load the saved Linear Regression model
with open('training1_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Function to predict EMISSION using the loaded model
def predict_species(fixed_acidity,volatile_acidity ,	citric_acid,	residual_sugar, chlorides,	free_sulfur_dioxide	,total_sulfur_dioxide,	density,	pH	,sulphates	,alcohol):
    features = np.array([fixed_acidity,volatile_acidity ,	citric_acid,	residual_sugar, chlorides,	free_sulfur_dioxide	,total_sulfur_dioxide,	density,	pH	,sulphates	,alcohol])
    features = features.reshape(1,-1)
    species = model.predict(features)
    return species[0]

# Streamlit UI
st.title('wine quality Prediction')
st.write("""
## Input Features
Enter the values for the input features to predict wine quality.
""")

# Input fields for user 
fixed_acidity = st.number_input('fixed_acidity')
volatile_acidity = st.number_input('volatile_acidity')
citric_acid = st.number_input('citric_acid')
residual_sugar = st.number_input('residual_sugar')
chlorides = st.number_input('chlorides')
free_sulfur_dioxide = st.number_input('free_sulfur_dioxide')
total_sulfur_dioxide = st.number_input('total_sulfur_dioxide')
density = st.number_input('density')
pH = st.number_input('pH')
sulphates = st.number_input('sulphates')
alcohol = st.number_input('alcohol')

# Prediction button
if st.button('Predict'):
    # Predict quality
    species_prediction = predict_species(fixed_acidity,volatile_acidity ,	citric_acid,	residual_sugar, chlorides,	free_sulfur_dioxide	,total_sulfur_dioxide,	density,	pH	,sulphates	,alcohol)
    st.write(f"Predicted quality: {species_prediction}")
