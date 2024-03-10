import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset
data = pd.read_excel('test/test2/your_dataset.xlsx')

# Separate features and target variable
X = data.drop(columns=['Freshness'])
y = data['Freshness']

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest classifier
model = RandomForestClassifier()

# Train the model
model.fit(X_train, y_train)

# Predict the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Save the model
joblib.dump(model, 'freshness_prediction_model.pkl')
