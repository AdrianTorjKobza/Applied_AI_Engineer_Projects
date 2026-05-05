import joblib
from sklearn.model_selection import train_test_split # Utility to split your dataset into a "training set" and a "testing set".
from sklearn.linear_model import LogisticRegression # Algorithm used for classification (predicting "yes/no" or "churn/no churn").
from sklearn.metrics import classification_report # Tool to measure how well the model performed (precision, recall, etc).
from preprocess import load_data, preprocess # Handles data loading and cleaning.

# Load the csv file.
df = load_data("../data/churn.csv")

# Preprocess.
# X represents the features (the data used to make a prediction).
# y represents the target (the actual outcome we want to predict).
X, y = preprocess(df)

# Save the feature names.
feature_columns = X.columns.tolist()
joblib.dump(feature_columns, "../models/features.pkl")

# Split the data.
# Reserves 20% of the data for testing and 80% for training.
# random_state=42: A "seed" that ensures the split is the same every time you run the code (useful for debugging).
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the model.
# Initializes the Logistic Regression algorithm. max_iter=1000 gives the model more time to "converge" or find the best mathematical fit.
model = LogisticRegression(max_iter=1000, class_weight="balanced")

# This is the actual training step. The model looks at the training features (X_train) and the known outcomes (y_train) to learn patterns.
model.fit(X_train, y_train)

# Evaluate.
# The model takes the "unseen" test features and makes its guesses.
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save the model.
# Serializes the trained model and saves it as a .pkl file. This allows you to use the model later in a web app or API without retraining it.
joblib.dump(model, "../models/model.pkl")
print("Model saved!")


