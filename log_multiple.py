import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
data = pd.read_csv('invoices(1).csv')

# Convert invoice_date to datetime and extract useful features
data['invoice_date'] = pd.to_datetime(data['invoice_date'], errors='coerce')
data['invoice_year'] = data['invoice_date'].dt.year
data['invoice_month'] = data['invoice_date'].dt.month

# Encode categorical features
data = pd.get_dummies(data, columns=['product_id', 'stock_code'], drop_first=True)

# Drop non-relevant columns
data.drop(['first_name', 'last_name', 'email', 'invoice_date', 'address', 'city', 'job'], axis=1, inplace=True)

# Handle missing values
data.fillna(0, inplace=True)

# Convert 'amount' into a binary classification target
data['high_value'] = (data['amount'] > 50).astype(int)

# Define features and target
X = data.drop(['amount', 'high_value'], axis=1)
y = data['high_value']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model training
model = LogisticRegression(max_iter=6000)
model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test)
print(f'Accuracy Score: {accuracy_score(y_test, y_pred):.2f}')
print(f'Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}')
print(f'Classification Report:\n{classification_report(y_test, y_pred)}')

# Function to predict multiple invoices
def predict_invoices():
    invoice_data = []
    n = int(input("Enter number of invoices: "))
    for i in range(n):
        qty = int(input(f"Enter qty for invoice {i + 1}: "))
        year = int(input(f"Enter invoice year for invoice {i + 1}: "))
        month = int(input(f"Enter invoice month for invoice {i + 1}: "))
        invoice_data.append({'qty': qty, 'invoice_year': year, 'invoice_month': month})

    new_invoices = pd.DataFrame(invoice_data)

    # Align new data columns with training data columns
    new_invoices = new_invoices.reindex(columns=X.columns, fill_value=0)

    # Ensure data types match
    new_invoices = new_invoices.astype(X.dtypes)

    # Prediction
    try:
        predictions = model.predict(new_invoices)
        print('Predictions (1 = High Value, 0 = Low Value):', predictions)
    except Exception as e:
        print(f"Error in prediction: {e}")

predict_invoices()
