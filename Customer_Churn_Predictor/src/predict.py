import joblib
import pandas as pd # Convert raw input data into a structured format (DataFrame) that the model understands

# Load the model.
model = joblib.load("../models/model.pkl")
feature_columns = joblib.load("../models/features.pkl")

def predict(input_data: dict):
    df = pd.DataFrame([input_data]) # Converts that single dictionary into a Pandas DataFrame (a row-and-column format).
    df = pd.get_dummies(df) # This performs One-Hot Encoding. It turns categorical text (like "Gender") into numbers (0s and 1s) so the math-based model can process it.

    # Align columns with training data
    df = df.reindex(columns=feature_columns, fill_value=0)

    # Feeds the processed data into the model to get a result.
    prediction = model.predict(df)
    return prediction[0]

# Example (IMPORTANT: must match training dataset!)
if __name__ == "__main__":
    sample = {
        "Account length": 100,
        "Area code": 415,
        "Customer service calls": 2,
        "Number vmail messages": 10,
        "International plan_Yes": 0
    }

    result = predict(sample)
    print("Churn Prediction:", result)


