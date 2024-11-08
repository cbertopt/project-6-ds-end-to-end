import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('trained_model.pkl')

# App title
st.title("Autism Screening Questionnaire")

# A1_Score to A10_Score questions
questions = [
    "Does your child look at you when you call his/her name?",
    "Does your child make unusual finger movements near his/her face?",
    "Does your child play pretend or make-believe?",
    "Does your child point with one finger to ask for something or to get help?",
    "Does your child stare at nothing with no apparent purpose?",
    "Does your child walk without tripping or falling?",
    "Does your child frequently repeat words or phrases?",
    "Does your child respond to your smile or other facial expressions?",
    "Does your child get easily upset by changes in routine?",
    "Does your child make eye contact with others?"
]

# Dictionary for storing answers (1 for YES, 0 for NO)
responses = []

# Variable to check if all questions have been answered
all_answered = True

# Display each question with a drop-down menu for answers, starting empty
for i, question in enumerate(questions, start=1):
    response = st.selectbox(question, ("Select an option", "YES", "NO"), key=f"Q{i}")
    if response == "Select an option":
        all_answered = False  # Mark as false if any question has not been answered
    elif response == "YES":
        responses.append(1)
    elif response == "NO":
        responses.append(0)

# Check that all the questions have been answered
if st.button("Check Autism Risk"):
    if not all_answered:
        st.warning("Please answer all the questions to see the results.")
    else:
        # Convert responses to a numpy array
        input_data = np.array([responses])
        
        try:
            # Make the prediction using the trained model
            prediction = model.predict(input_data)
            risk_prob = model.predict_proba(input_data)[:, 1]

            # Display the result
            if prediction[0] == 1:
                st.write(f"The autism risk is **HIGH** (Risk Probability: {risk_prob[0]:.2f}).")
                st.write(
                    """
                    If your child shows signs of autism, it is important to consult a healthcare provider for an early assessment. 
                    Early intervention can make a significant difference.

                    - **Contact your nearest health center** to schedule a consultation with a specialist.
                    - You can also reach out to the **National Health Service (SNS)** in Portugal for guidance: 
                      [SNS Website](https://www.sns.gov.pt/).
                    - For more information on autism and early signs, visit [Autism Speaks](https://www.autismspeaks.org/).

                    Remember, professional guidance is essential to provide the best support for your child.
                    """
                )
            else:
                st.write(f"The autism risk is **LOW** (Risk Probability: {risk_prob[0]:.2f}).")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")