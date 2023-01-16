import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from time import time
from sklearn.metrics import f1_score



data_frame = []
data_files = ['2021-2022.csv', '2020-2021.csv', '2019-2020.csv']

for data_file in data_files:
    data_frame.append(pd.read_csv(data_file))

data = pd.concat(data_frame).reset_index()

encoder = LabelEncoder()
home_team_enc = encoder.fit_transform(data['HomeTeam'])
home_encoded_mapping = dict(
    zip(encoder.classes_, encoder.transform(encoder.classes_).tolist()))
data['home_team_enc'] = home_team_enc

encoder = LabelEncoder()
away_team_enc = encoder.fit_transform(data['AwayTeam'])
away_team_enc_mapping = dict(
    zip(encoder.classes_, encoder.transform(encoder.classes_).tolist()))
data['away_team_enc'] = away_team_enc
#print(data.head(5))
#print(data[data.isna().any(axis=1)]) # 5 rows
#data = data.dropna(axis=0)
#print(data.head(5))


#remove = ['Div','Date','Time','HomeTeam','AwayTeam','HTR','Referee','B365>2.5','B365<2.5','P>2.5','P<2.5','Max>2.5','Max<2.5','Avg>2.5','AHh','B365AHH','B365AHA','PAHH','PAHA','MaxAHH','MaxAHA','AvgAHH']
#for x in remove:
    #del data[x]
#print(data['index'].head(5))

features = ['index','home_team_enc','away_team_enc','HTHG','HTAG','HS','AS','HST','AST','HR','AR']
label = ['FTR']
all_data = features + label
data = data[all_data]
#del data[data. columns[0]] 
""" data = data.astype({'FTR':'string'})
data['FTR'].replace('A', 0,inplace=True)
data['FTR'].replace('D', 1,inplace=True)
data['FTR'].replace('H', 2,inplace=True)
 """

upper_case =['HTHG','HTAG','HS','AS','HST', 'AST','HR','AR','FTR']
lower_case=['index','home_team_enc','away_team_enc','hthg', 'htag','hs','as','hst','ast','hr','ar']

for name in upper_case:
    data.rename(columns={name: name.lower()}, inplace=True)

#print(data.dtypes)



#print(data.head(5))


X_features = data[lower_case]
X_features = X_features.dropna(axis=0)

#print(X_features.head())
y_label = data['ftr']

del data['as']
del data['index']
data['index'] = data.index
data.to_csv('premier_league_dataset.csv', index=False)
#print(y_label.head())

del data['ftr']

#y_label['FTR'] = y_label['FTR'].astype("string")
#y_label['FTR'].replace('A', 0,inplace=True)
#y_label['FTR'].replace('D', 1,inplace=True)
#y_label['FTR'].replace('H', 2,inplace=True)
#print(X_features.head(20))
def train_classifier(clf, X_train, y_train):
    start = time()
    clf.fit(X_train, y_train)
    end = time()
    print("Model trained in {:2f} seconds".format(end-start))


def predict_labels(clf, features, target):
    start = time()
    y_pred = clf.predict(features)
    end = time()
    print("Made Predictions in {:2f} seconds".format(end-start))

    acc = sum(target == y_pred) / float(len(y_pred))

    return f1_score(target, y_pred, average='micro'), acc
#heatmap = sns.heatmap(X_features.corr()[["FTR"]].sort_values(by="FTR", ascending=False), vmin=-1, vmax=1, annot=True, cmap="BrBG")
#heatmap.set_title("Features Correlating with Survived", fontdict={"fontsize":18}, pad=16)
#print(X_features.head())

def model(clf, X_train, y_train, X_test, y_test):
    train_classifier(clf, X_train, y_train)

    f1, acc = predict_labels(clf, X_train, y_train)
    print("Training Info:")
    print("-" * 20)
    print("F1 Score:{}".format(f1))
    print("Accuracy:{}".format(acc))

    f1, acc = predict_labels(clf, X_test, y_test)
    print("Test Metrics:")
    print("-" * 20)
    print("F1 Score:{}".format(f1))
    print("Accuracy:{}".format(acc))
X_train, X_test, Y_train, Y_test = train_test_split(X_features, y_label, test_size=0.2)
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
#independent columns
   #target column i.e price range
#apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(X_features,y_label)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X_features.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
#print(featureScores.nlargest(10,'Score')) 
 #print 10 best features
rfClassifier = RandomForestClassifier()
model(rfClassifier, X_train, Y_train, X_test, Y_test)
