import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv('invoices.csv')

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

# Define features and target
X = data.drop('amount', axis=1)
y = data['amount']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test)
print(f'Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}')  #mean_squared_error(): Measures the average squared difference between actual (y_test) and predicted (y_pred) values.
print(f'R-squared Score: {r2_score(y_test, y_pred):.2f}')            # r2_score(): Indicates how well the model fits the data. Values close to 1 mean better performance.


# Create new invoice data with all encoded columns set to 0
new_invoice = pd.DataFrame(0, index=[0], columns=X.columns)

# Set relevant features for prediction
new_invoice['qty'] = 7
new_invoice['invoice_year'] = 2023
new_invoice['invoice_month'] = 52

# Ensure encoded columns are updated correctly
if 'product_id_133' in new_invoice.columns:
    new_invoice['product_id_133'] = 1
if 'stock_code_36239634' in new_invoice.columns:
    new_invoice['stock_code_36239634'] = 1

# Ensure data types match
new_invoice = new_invoice.astype(X.dtypes)

# Prediction
try:
    predicted_amount = model.predict(new_invoice)
    print(f'Predicted Amount: {predicted_amount[0]:.2f}')
except Exception as e:
    print(f"Error in prediction: {e}")

# Align `new_invoice` columns dynamically
new_invoice = pd.DataFrame(0, index=[0], columns=X.columns).combine_first(new_invoice)


