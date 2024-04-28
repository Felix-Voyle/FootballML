import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


data = pd.read_csv('data/data.csv')

# Create 'result' column based on goal counts
data['result'] = data.apply(lambda row: 1 if row['home_team_goal_count'] > row['away_team_goal_count'] else (-1 if row['home_team_goal_count'] < row['away_team_goal_count'] else 0), axis=1)


X = data[['home_team_name', 'away_team_name']]
y = data['result']

# One-hot encoding
X = pd.get_dummies(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print("Classification Accuracy:", accuracy)

# Prediction on new data
remaining_matches = pd.read_csv('data/remaining_matches.csv')
remaining_matches_encoded = pd.get_dummies(remaining_matches[['home_team_name', 'away_team_name']])
remaining_predictions = model.predict(remaining_matches_encoded)

print("Predicted outcomes for remaining matches:")
for index, prediction in enumerate(remaining_predictions):
    home_team = remaining_matches.iloc[index]['home_team_name']
    away_team = remaining_matches.iloc[index]['away_team_name']
    prediction_label = "Home Team Wins" if prediction == 1 else ("Away Team Wins" if prediction == -1 else "Draw")
    print(f"{home_team} vs {away_team}: {prediction_label}")

