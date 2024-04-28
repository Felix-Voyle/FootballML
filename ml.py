import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('data/data.csv')

# Create a new column 'result' based on 'home_team_goal_count' and 'away_team_goal_count'
data['result'] = data.apply(lambda row: 1 if row['home_team_goal_count'] > row['away_team_goal_count'] else (-1 if row['home_team_goal_count'] < row['away_team_goal_count'] else 0), axis=1)

# Calculate total shots on target and total games played for home teams
home_shots_on_target = data.groupby('home_team_name')['home_team_shots_on_target'].sum()
home_games_played = data['home_team_name'].value_counts()

# Calculate total shots on target and total games played for away teams
away_shots_on_target = data.groupby('away_team_name')['away_team_shots_on_target'].sum()
away_games_played = data['away_team_name'].value_counts()

# Calculate average shots on target per game for home and away teams
average_shots_on_target_home = home_shots_on_target / home_games_played
average_shots_on_target_away = away_shots_on_target / away_games_played

# Assign calculated averages to the DataFrame
data['average_shots_on_target_home'] = data['home_team_name'].map(average_shots_on_target_home)
data['average_shots_on_target_away'] = data['away_team_name'].map(average_shots_on_target_away)

# Select relevant columns for classification
relevant_columns_class = ['home_team_name', 'away_team_name', 'result',  'average_shots_on_target_home', 'average_shots_on_target_away']
if 'average_shots_on_target_home' in data.columns and 'average_shots_on_target_away' in data.columns:
    relevant_columns_class += ['average_shots_on_target_home', 'average_shots_on_target_away']

data_class = data[relevant_columns_class]

# One-hot encode team names
data_class = pd.get_dummies(data_class, columns=['home_team_name', 'away_team_name'])

# Define features and target variables for classification
X_class = data_class.drop(columns=['result'])
y_class = data_class['result']

# Split the data into training and testing sets for classification
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_class, y_class, test_size=0.2, random_state=42, shuffle=True)

# Define the hyperparameters grid for classification
param_grid_class = {
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
}

# Instantiate the Decision Tree classifier
dt_classifier = DecisionTreeClassifier(random_state=42)

# Instantiate the GridSearchCV object for classification
grid_search_class = GridSearchCV(dt_classifier, param_grid_class, cv=5, scoring='accuracy')

# Fit the grid search to the data for classification
grid_search_class.fit(X_train_class, y_train_class)

# Get the best hyperparameters for classification
best_params_class = grid_search_class.best_params_

print("Best Hyperparameters for Classification:", best_params_class)

# Make predictions using the best estimator for classification
best_estimator_class = grid_search_class.best_estimator_
predictions_class = best_estimator_class.predict(X_test_class)

# Evaluate the model
accuracy = accuracy_score(y_test_class, predictions_class)
print("Classification Accuracy:", accuracy)

# Prepare data for remaining matches
remaining_matches = pd.read_csv('data/remaining_matches.csv')

# Calculate average shots on target per game for home and away teams for the historical data
if 'average_shots_on_target_home' in data.columns and 'average_shots_on_target_away' in data.columns:
    average_shots_on_target_home = data.groupby('home_team_name')['home_team_shots_on_target'].mean()
    average_shots_on_target_away = data.groupby('away_team_name')['away_team_shots_on_target'].mean()

    # Map the averages to the remaining matches based on team names
    remaining_matches['average_shots_on_target_home'] = remaining_matches['home_team_name'].map(average_shots_on_target_home)
    remaining_matches['average_shots_on_target_away'] = remaining_matches['away_team_name'].map(average_shots_on_target_away)

    # Fill NaN values with the overall mean shots on target for each team
    overall_mean_shots_on_target_home = data['home_team_shots_on_target'].mean()
    overall_mean_shots_on_target_away = data['away_team_shots_on_target'].mean()
    
    remaining_matches['average_shots_on_target_home'] = remaining_matches['average_shots_on_target_home'].fillna(overall_mean_shots_on_target_home)
    remaining_matches['average_shots_on_target_away'] = remaining_matches['average_shots_on_target_away'].fillna(overall_mean_shots_on_target_away)

    # Select relevant columns for remaining matches and ensure the same order of columns as in the training data
    relevant_columns_remaining = ['home_team_name', 'away_team_name', 'average_shots_on_target_home', 'average_shots_on_target_away']
    remaining_matches_processed = remaining_matches[relevant_columns_remaining]

    # Ensure that the order of columns in remaining_matches_processed matches the order during training
    X_train_columns = list(X_train_class.columns)
    remaining_matches_processed = remaining_matches_processed.reindex(columns=X_train_columns, fill_value=0)

    # Make predictions for remaining matches using classification model
    remaining_predictions_class = best_estimator_class.predict(remaining_matches_processed)

    # Print predictions
    print("Predicted outcomes for remaining matches:")
    for index, prediction in enumerate(remaining_predictions_class):
        home_team = remaining_matches.iloc[index]['home_team_name']
        away_team = remaining_matches.iloc[index]['away_team_name']
        prediction_label = "Home Team Wins" if prediction == 1 else ("Away Team Wins" if prediction == -1 else "Draw")
        print(f"{home_team} vs {away_team}: {prediction_label}")
else:
    print("Some relevant columns are missing in the historical data. Unable to make predictions.")

