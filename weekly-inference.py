import os 
import modal

LOCAL = True

if LOCAL == False:
    stub = modal.Stub()
    image = modal.Image.debian_slim().apt_install(["libgomp1"]).pip_install(["hopsworks==3.0.4", "seaborn", "joblib", "scikit-learn"])

    @stub.function(image=image, schedule=modal.Period(days=8), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
    def f():
        g()


def g():
    import hopsworks
    import pandas as pd
    import random
    from sklearn.preprocessing import LabelEncoder

 
    project = hopsworks.login()
    fs = project.get_feature_store()

    latest_csv = pd.read_csv("https://www.football-data.co.uk/mmz4281/2223/E0.csv")
    latest_entry = latest_csv.tail(1)

    #latest_entry = latest_csv.iloc[-1]

    print(latest_entry)
    # features = ['HomeTeam','AwayTeam','HTHG', 'HTAG','HS','HST','AST','HR','AR']
    features_inference = pd.DataFrame()
    #features_inference_list = ['HomeTeam','AwayTeam','HTHG', 'HTAG','HS','HST','AST','HR','AR']
    features_inference_list = [{'hthg': latest_entry.iloc[0,8], 'htag': latest_entry.iloc[0,9], 'hs':latest_entry.iloc[0,12], 'hst': latest_entry.iloc[0,14],'ast': latest_entry.iloc[0,15],'hr': latest_entry.iloc[0,22],
    'ar': latest_entry.iloc[0,23], 'result': latest_entry.iloc[0,7]}]

    features_inference = pd.DataFrame(features_inference_list)

    def transformResult(row):
        if(row.result == 'H'):
            return 1
        elif(row.result == 'A'):
            return -1
        else:
            return 0
    features_inference["result"] = features_inference.apply(lambda row: transformResult(row),axis=1)
    features_inference['index'] = round(random.uniform(1150,3000))

    print(features_inference.dtypes)

    premier_league_fg = fs.get_or_create_feature_group(
        name="premier_league",
        version=6,
        primary_key=["index"], 
        description="Premier League Predictor")  
    premier_league_fg.insert(features_inference, write_options={"wait_for_job" : False})

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()