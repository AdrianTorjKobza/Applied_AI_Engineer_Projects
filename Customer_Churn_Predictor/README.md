
# Customer Churn Predictor

This project is a machine learning model that predicts whether a customer will churn (leave a service) based on historical customer data.

It demonstrates a basic end-to-end ML workflow:
- Data loading and preprocessing
- Feature engineering using pandas
- Model training using Logistic Regression
- Model evaluation
- Saving and loading trained model
- Making predictions on new data

---

## Tech Stack

- Python 3.x  
- Pandas  
- NumPy  
- Scikit-learn  
- Joblib  

---

## Dataset

The dataset contains customer information such as:
- Tenure  
- Monthly charges  
- Support calls  
- Contract type  
- Churn label (target variable)  

Target variable:
- `churn = 1` → Customer left  
- `churn = 0` → Customer stayed  

---

## Machine Learning Approach

### Model Used:
- Logistic Regression

### Workflow:
1. Load dataset  
2. Clean and preprocess data  
3. Encode categorical variables using `get_dummies`  
4. Split dataset into train/test sets  
5. Train model  
6. Evaluate performance  
7. Save trained model  

---

# How to Run the Project

---

## Step 1: Install dependencies

```bash
pip install -r requirements.txt
```

---

## Step 2: Train the model

Run the training script to train and save the model:

```bash
cd src
python train.py
```

### What happens in this step:
- Loads dataset
- Applies preprocessing
- Splits data into training and testing sets
- Trains Logistic Regression model
- Evaluates performance (precision, recall, F1-score)
- Saves model to:

```
models/model.pkl
```

---

## Step 3: Make predictions

Run prediction script:

```bash
python predict.py
```

### Example input:

```python
sample = {
    "tenure": 12,
    "monthly_charges": 70,
    "support_calls": 2
}
```

### Output:

```
Churn Prediction: 1
```

- `1` → Customer will churn/leave  
- `0` → Customer will stay  

---

## Model Evaluation

The model is evaluated using:

- Accuracy  
- Precision  
- Recall  
- F1-score  

Example output:

```
precision    recall  f1-score   support

False       0.94      0.73      0.82
True        0.22      0.60      0.32

accuracy    0.72
```
