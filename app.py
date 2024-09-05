import streamlit as st
import numpy as np
import joblib
import os

st.title("Bike Purchase Predictor")

st.divider()

# Input fields
income = st.number_input(
    "Enter your income",
    min_value=0,
    value=300000
)

age = st.number_input(
    "Enter your age",
    min_value=15,
    value=30
)

education = st.number_input(
    "Enter your education level (0 to 2)",
    min_value=0, max_value=2, value=2
)

homeowner = st.number_input(
    "Are you a homeowner? (0 for No, 1 for Yes)",
    min_value=0, max_value=1, value=1
)


model_path = "model.pkl"

if not os.path.isfile(model_path):
    st.error(
        "Model file not found. Ensure 'model.pkl' is in the correct directory."
    )
    st.stop()

# Load the model
try:
    model = joblib.load(model_path)
except FileNotFoundError:
    st.error(
        "Model file not found. Ensure 'model.pkl' is in the correct directory."
    )
    st.stop()
except Exception as e:
    st.error(
        f"An error occurred while loading the model: {e}"
    )
    st.stop()

# Prepare the input for prediction
X = [income, age, education, homeowner]

st.divider()

# Prediction button
predictbutton = st.button("Press for prediction of bike purchase")

st.divider()

if predictbutton:
    st.balloons()

    # Convert input to numpy array
    X1 = np.array([X])

    try:
        # Make prediction
        prediction = model.predict(X1)[0]
        result = "Customer will purchase" if prediction == 1 else "Customer won't buy"
        st.write(result)
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
else:
    st.write("Please enter values and use the predict button")





























# import streamlit as st
# import numpy as np
# import joblib

# st.title("Bike Purchase Predictor")

# st.divider()

# income = st.number_input("Enter your income", min_value=0, value=300000)
# age = st.number_input("Enter your age", min_value=15, value=30)
# education = st.number_input("Enter your education", min_value=0, value=2)
# homeowner = st.number_input("Home owner input", min_value=0, value=1)

# model = joblib.load("model.pkl")

# X = [income, age, education, homeowner]

# st.divider()

# predictbutton = st.button("Press for prediction of bike purchase")

# st.divider()

# if predictbutton:

#     st.balloons()

#     X1 = np.array([X])

#     prediction = model.predict(X1)[0]

#     result = "Customer will purchase" if prediction == 1 else "Customer won't buy"
   
#     st.write(result)


# else:
#     st.write("Please enter values and use predict button")
















#Index(['Income', 'Age', 'Education', 'Home Owner'], dtype='object')