from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import os

app = Flask(__name__)

df = pd.read_csv(r"E:\Projects\MBA in Finance\Datset.csv")

Prod_type_Mapping = {
    'Insurance': 1,
    'Personal Loan': 2,
    'Home Loan': 3,
    'Savings Account': 4,
    'Credit Card': 5,
    'No Loan': 6,
    'Car Loan': 7,
    'Student Loan': 8
}

def replace(ProductType):
    return Prod_type_Mapping.get(ProductType, ProductType)

df['Product Type'] = df['Product Type'].apply(replace)
df['Loan History'] = df['Loan History'].apply(replace)

for col in ['Income', 'Credit Score', 'Amount', 'Age']:
    df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

x = df[['Age', 'Income', 'Credit Score', 'Loan History']]
y = df['Product Type']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [100, 150, 200],
    'max_depth': [10, 15, 20],
    'min_samples_split': [2, 5]
}

randomized_search = RandomizedSearchCV(RandomForestClassifier(class_weight="balanced", random_state=42),
                                       param_distributions=param_grid, n_iter=10, cv=5, random_state=42, n_jobs=-1)
randomized_search.fit(x_train, y_train)
best_model = randomized_search.best_estimator_

classification_report_dict = classification_report(y_test, best_model.predict(x_test), output_dict=True)
conf_matrix = confusion_matrix(y_test, best_model.predict(x_test))

@app.route('/')
def home():
    accuracy = round(classification_report_dict['accuracy'] * 100, 2)
    return render_template('index.html', accuracy=accuracy)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        age = float(data.get('age', 0))
        income = float(data.get('income', 0))
        credit_score = float(data.get('credit_score', 0))
        loan_history = Prod_type_Mapping.get(data.get('loan_history', 'No Loan'), 6)

        # making for input data
        input_data = pd.DataFrame([[age, income, credit_score, loan_history]], 
                                  columns=['Age', 'Income', 'Credit Score', 'Loan History'])

        # Normalization
        for col in ['Income', 'Credit Score', 'Age']:
            input_data[col] = (input_data[col] - df[col].min()) / (df[col].max() - df[col].min())

        
        prediction = best_model.predict(input_data)
        predicted_product = next((key for key, value in Prod_type_Mapping.items() if value == prediction[0]), "Unknown")

        
        probabilities = best_model.predict_proba(input_data)[0]
        prediction_probabilities = {key: prob for key, prob in zip(Prod_type_Mapping.keys(), probabilities)}

        return jsonify({
            'predicted_product': predicted_product,
            'probabilities': prediction_probabilities
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/metrics', methods=['GET'])
def metrics():
    predictions = best_model.predict(x_test)
    prediction_counts = pd.Series(predictions).value_counts().to_dict()
    prediction_distribution = {key: prediction_counts.get(value, 0) for key, value in Prod_type_Mapping.items()}

    return jsonify({
        'accuracy': round(classification_report_dict['accuracy'] * 100, 2),
        'confusion_matrix': conf_matrix.tolist(),
        'prediction_distribution': prediction_distribution
    })

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5001)
