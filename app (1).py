
import pandas as pd
import pickle

# Load the trained Random Forest Regressor model
with open('random_forest_regressor_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the label encoders
with open('label_encoders.pkl', 'rb') as file:
    label_encoders = pickle.load(file)

def predict_salary(age, gender, education_level, job_title, years_of_experience):
    # Create a DataFrame for the input
    input_data = pd.DataFrame([{
        'Age': age,
        'Gender': gender,
        'Education Level': education_level,
        'Job Title': job_title,
        'Years of Experience': years_of_experience
    }])

    # Apply label encoding to categorical features using the loaded encoders
    # Handle cases where a new category might appear (though less likely with fixed input)
    for col, le in label_encoders.items():
        # Use transform, or fit_transform if new category is expected and should be handled
        # For a deployed app, it's safer to ensure all categories are known by the encoder.
        # Here, we assume the input categories are known from training data.
        # If a category is new, it will raise an error unless handled, e.g., using a try-except or mapping unknown to a default.
        input_data[col] = le.transform(input_data[col])

    # Make prediction
    prediction = model.predict(input_data)
    return prediction[0]

if __name__ == '__main__':
    # Example usage:
    # Note: 'Gender', 'Education Level', 'Job Title' need to be passed as their original string values
    # as they were in the training data, so the label encoder can transform them.

    # Example 1: Male, Bachelor's, Software Engineer, 5 years experience, 32 years old
    salary1 = predict_salary(age=32.0, gender='Male', education_level='Bachelor's', job_title='Software Engineer', years_of_experience=5.0)
    print(f"Predicted Salary for Example 1: ${salary1:.2f}")

    # Example 2: Female, Master's, Data Analyst, 3 years experience, 28 years old
    salary2 = predict_salary(age=28.0, gender='Female', education_level='Master's', job_title='Data Analyst', years_of_experience=3.0)
    print(f"Predicted Salary for Example 2: ${salary2:.2f}")

    # Example 3: Male, PhD, Senior Manager, 15 years experience, 45 years old
    salary3 = predict_salary(age=45.0, gender='Male', education_level='PhD', job_title='Senior Manager', years_of_experience=15.0)
    print(f"Predicted Salary for Example 3: ${salary3:.2f}")
