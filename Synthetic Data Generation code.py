from faker import Faker
import pandas as pd
import random

# Initialize Faker
fake = Faker()

# Function to generate a single customer's data
def generate_customer_data():
    # Generate random customer demographics
    customer_id = fake.uuid4()
    name = fake.name()
    age = random.randint(18, 70)  # Random age between 18 and 70
    income = round(random.uniform(20000, 150000), 2)  # Random income between 20,000 and 150,000
    credit_score = random.randint(300, 850)  # Random credit score between 300 and 850
    loan_history = random.choice(['No Loan', 'Home Loan', 'Personal Loan', 'Car Loan', 'Student Loan'])

    # Generate random transaction data
    transactions = []
    for _ in range(random.randint(1, 5)):  # Random number of transactions (1 to 5)
        product_type = random.choice(['Personal Loan', 'Home Loan', 'Credit Card', 'Savings Account', 'Insurance'])
        amount = round(random.uniform(100, 5000), 2)  # Random transaction amount
        transaction_date = fake.date_this_year()  # Random date this year
        transaction_frequency = random.choice(['Once', 'Weekly', 'Monthly', 'Yearly'])  # Random frequency
        transactions.append([customer_id, name, product_type, amount, transaction_date, transaction_frequency])
    
    return age, income, credit_score, loan_history, transactions

# Generate data for 10,000 customers
combined_data = []

for _ in range(10000):  # Generate data for 10,000 customers
    age, income, credit_score, loan_history, transactions = generate_customer_data()
    
    # For each transaction, add customer demographic details to it
    for transaction in transactions:
        combined_data.append([age, income, credit_score, loan_history] + transaction)

# Create pandas DataFrame with combined data
combined_df = pd.DataFrame(combined_data, columns=[
    'Age', 'Income', 'Credit Score', 'Loan History',
    'Customer ID', 'Name', 'Product Type', 'Amount', 'Transaction Date', 'Transaction Frequency'
])

# Print the first few rows of the combined DataFrame to verify
print("Combined Data:")
print(combined_df.head())

# Save the combined data to Excel
combined_df.to_excel('synthetic_customer_data_10000_combined.xlsx', index=False)

print("\nData has been saved to 'synthetic_customer_data_10000_combined.xlsx'")
