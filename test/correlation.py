import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load data from Excel file
df = pd.read_excel("test/your_dataset.xlsx")  # Replace "your_dataset.xlsx" with the path to your Excel file

# Split data into features and target
X = df.drop('Freshness', axis=1)
y = df['Freshness']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train Random Forest classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Get all unique labels in the dataset
labels = np.unique(np.concatenate((y_test, y_pred)))

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Display classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Display confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=labels))

# Input a specific set of features for prediction
input_features = [[0.16, 0.08, 6.7, 48]]  # Replace these values with your specific features

# Predict the freshness label for the input features
predicted_freshness = clf.predict(input_features)
print("\nPredicted Freshness:", predicted_freshness)

# Exclude non-numeric columns from correlation calculation
numeric_cols = df.select_dtypes(include=[np.number]).columns
corr = df[numeric_cols].corr()

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()
