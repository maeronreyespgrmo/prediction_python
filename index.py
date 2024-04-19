# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Sample historical data (you should replace this with your actual data)
data = {'TeamA_goals': [2, 3, 1, 2, 1],
        'TeamA_possession': [60, 55, 63, 58, 59],
        'TeamA_shots': [8, 7, 5, 9, 6],
        'TeamB_goals': [1, 2, 0, 1, 3],
        'TeamB_possession': [40, 45, 37, 42, 41],
        'TeamB_shots': [4, 6, 3, 5, 7],
        'Winner': ['TeamA', 'TeamA', 'TeamA', 'TeamA', 'TeamB']}

df = pd.DataFrame(data)

# Define features and target
X = df[['TeamA_goals', 'TeamA_possession', 'TeamA_shots', 'TeamB_goals', 'TeamB_possession', 'TeamB_shots']]
y = df['Winner']

# Encode the target variable
le = LabelEncoder()
y = le.fit_transform(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict the winners
predictions = model.predict(X_test)

# Round the predictions to the nearest integer
predictions = [round(pred) for pred in predictions]

# Calculate the accuracy score
accuracy = accuracy_score(y_test, predictions)

print("Accuracy of winner prediction using linear regression:", accuracy)

# Find the winner based on predictions
predicted_winners = ['TeamA' if pred > 0 else 'TeamB' for pred in predictions]

for i, pred_winner in enumerate(predicted_winners):
    print(f"Predicted Winner for Test {i+1}: {pred_winner}")