# Smart Loan Product Advisor Personalized Financial Products Recommendations

This project provides an intelligent system for predicting personalized loan products based on user financial details. By leveraging machine learning, it predicts the most suitable loan product for users based on their age, income, credit score, and loan history. The goal is to make the loan product selection process more efficient, tailored, and easy for customers.

## Features
- **Predict Loan Products**: The model predicts personalized loan products based on user input.
- **Model Accuracy**: Displays the performance of the model, including accuracy and confusion matrix.
- **Interactive Dashboard**: An easy-to-use web interface where users can input their financial details to get predictions.
- **Chart Visualization**: Shows the prediction probabilities for different loan products in a bar chart.

## Dataset Information
The dataset used in this project is generated using the **Faker** library due to the lack of a real-world dataset. As a result, the prediction outputs are based on simulated data and may not accurately represent real-world scenarios. However, the predictions are generated from the machine learning model's learning based on the simulated data.

**Note**: Since the dataset is synthetic, the predictions shown by the system may not always be appropriate for real-world financial decisions. 

## Technologies Used
- **Backend**: Flask (Python framework)
- **Machine Learning**: Scikit-learn (Random Forest Classifier)
- **Frontend**: HTML, CSS, JavaScript 
- **Data Processing**: Pandas, NumPy
- **Deployment**: Local environment or any server that supports Flask

## Usage

1. Enter your financial details (Age, Monthly Income, Credit Score, and Loan History) in the input form.
2. Click on the "Predict" button to receive a personalized loan product recommendation.
3. View the prediction probabilities in the chart for each loan product type.

## Contributing

Contributions are welcome! Feel free to fork the repository, open an issue, or submit a pull request. Ensure that you follow the project's code of conduct and contribute in a way that is respectful to others.


