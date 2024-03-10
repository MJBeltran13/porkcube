import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
data = pd.read_excel("test\your_dataset.xlsx")

# Define feature names
feature_names = ['Ammonia', 'Methane', 'pH_level', 'Lightness_L']

# Assign feature names to columns
data.columns = feature_names + ['Freshness']

# Split data into features and target variable
X = data[feature_names]
y = data['Freshness']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the logistic regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Function to plot decision boundary
def plot_decision_boundary(X, y, model, feature_names, ax):
    # Set min and max values for the features
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    # Generate a grid of points
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
    # Predict the labels for each point in the grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel(), np.zeros_like(xx.ravel()), np.zeros_like(xx.ravel())])  # Include all features here
    Z = Z.reshape(xx.shape)
    
    # Mask invalid values
    Z = np.ma.masked_invalid(Z)
    
    # Plot the decision boundary
    ax.contourf(xx, yy, Z, alpha=0.3)
    
    # Plot class samples
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])

# Create a figure and axis
fig, ax = plt.subplots(figsize=(8, 6))

# Plot decision boundary using all features
plot_decision_boundary(X_test.values, y_test, model, feature_names, ax)

# Set title
ax.set_title('Decision Boundary for Logistic Regression Model')

# Show the plot
plt.show()

# Now, let's predict the freshness of a new data point
new_data = [[0.3, 0.11, 6.6, 45]]  # Assuming the same order of features

# Make prediction
prediction = model.predict(new_data)

# Print the prediction
print("Prediction for the input data:", prediction)
