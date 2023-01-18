# MatchPrediction

The purpose of this project was to create a scalable prediction service for premier league football matches predicions.
The prediction for the result of the game (Home Team Win, Draw, Away Team Win) is based on half time statistics.

The data were collected from the page https://www.football-data.co.uk. Last three season were considered (2019-2020, 2020-2021, 2021-2022).
The available features are shown below.

<img width="536" alt="Screenshot 2023-01-18 at 13 16 42" src="https://user-images.githubusercontent.com/115500459/213169389-ea68d282-3ba0-4ee2-ae8e-de47d6a04ecc.png">
To determine the features can contributed the most to the prediction SelectKBest was applied to the data and the best 10 features were considered.

<img width="220" alt="Screenshot 2023-01-16 at 16 34 55" src="https://user-images.githubusercontent.com/115500459/213170042-c6d00dc8-f901-44b8-98c8-c5babf622863.png">

Next step was to create a feature pipeline where a feature group is created in hopsworks and the data are uploaded.
For the training RandomForestClassifier model was used with min_samples_split=6 and max_depth=6 with a 0.62 accuracy.

An inference program was created that used the last entry of the last season everyweek to create a new entry, with the features that were considered for the training, for the feature group in hopsworks.

The batch inference program makes a prediction based on the last entry in the feature group and compares it with the real result of the match. It then creates a confusion matrix with all the previous results. In addition a new feature group is created in hopwsworks with the features data, prediction and label.

Future improvements:

In this implementation the players of the teams are not considered, the stats of the football players of the diffeent teams could improve the performance of the model. An another improvement could be the creation of features such as average goals scored etc. 
In this way the would have an idea of the overall performance of the team for a speficic season.

