import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import learning_curve
import numpy as np

# Read data from Excel file
df = pd.read_excel("test\your_dataset.xlsx")

# 1. Data Distribution
df.hist(figsize=(10, 8))
plt.tight_layout()
plt.show()

# 2. Correlation Matrix
df_numerical = df.drop(columns=['Freshness'])
corr_matrix = df_numerical.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

# 3. Decision Boundaries (for two features)
X = df[['Ammonia', 'Methane']]
if len(X.columns) == 2:
    model = DecisionTreeClassifier()
    model.fit(X, df['Freshness'].map({'Fresh': 1, 'Not Fresh': 0}))

    # Plot decision boundaries
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='Ammonia', y='Methane', hue='Freshness', data=df, palette='viridis')
    xx, yy = np.meshgrid(np.linspace(df['Ammonia'].min(), df['Ammonia'].max(), 100),
                         np.linspace(df['Methane'].min(), df['Methane'].max(), 100))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    plt.xlabel('Ammonia')
    plt.ylabel('Methane')
    plt.title('Decision Boundaries')
    plt.show()

# Split the dataset into features (X) and target labels (y)
X = df[['Ammonia', 'Methane', 'pH_level', 'Lightness_L']]
y = df['Freshness'].map({'Fresh': 1, 'Not Fresh': 0})  # Convert labels to binary

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Confusion Matrix
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', cbar=False)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# 5. Feature Importance
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
feature_importance = model.feature_importances_
plt.bar(X.columns, feature_importance)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance')
plt.show()

# 6. ROC Curve
y_probs = model.predict_proba(X_test)[:, 1]  # Probability predictions for positive class
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

# 7. Learning Curve
train_sizes, train_scores, test_scores = learning_curve(DecisionTreeClassifier(), X, y, cv=3, scoring='accuracy')
train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
plt.plot(train_sizes, train_scores_mean, label='Training Accuracy')
plt.plot(train_sizes, test_scores_mean, label='Validation Accuracy')
plt.xlabel('Training Examples')
plt.ylabel('Accuracy')
plt.title('Learning Curve')
plt.legend()
plt.show()
