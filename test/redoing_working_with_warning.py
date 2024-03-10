import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_excel("test\your_dataset.xlsx")

# Explicitly set feature names
feature_names = ['Ammonia', 'Methane', 'pH_level', 'Lightness_L']

# Rename columns
data.columns = feature_names + ['Freshness']

# Split data into features and target variable
X = data.drop(columns=['Freshness'])
y = data['Freshness']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the logistic regression model with explicit feature names
model = LogisticRegression(max_iter=1000, multi_class='auto', solver='lbfgs', penalty='none', random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Now, let's predict the freshness of a new data point
new_data = [[0.15, 0.08, 6.7, 48]]

# Make prediction
prediction = model.predict(new_data)

# Print the prediction
print("Prediction for the input data:", prediction)
