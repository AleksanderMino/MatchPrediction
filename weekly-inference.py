import os 
import modal

BACKFILL = False
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
    from sklearn.preprocessing import LabelEncoder


    project = hopsworks.login()
    fs = project.get_feature_store()

    latest_csv = pd.read_csv("https://www.football-data.co.uk/mmz4281/2223/E0.csv")
    latest_entry = latest_csv.tail(1)
    print(latest_entry)
    features = ['HomeTeam','AwayTeam','HTHG', 'HTAG','HS','HST','AST','HR','AR']
    features_inference = pd.DataFrame()
    features_inference_list = ['HomeTeam','AwayTeam','HTHG', 'HTAG','HS','HST','AST','HR','AR']
    features_inference_list = [{'HomeTeam': latest_entry.iloc[0,3], 'AwayTeam': latest_entry.iloc[0,4] ,'HTHG': latest_entry.iloc[0,8], 
    'HTAG': latest_entry.iloc[0,9], 'HS':latest_entry.iloc[0,12], 'HST': latest_entry.iloc[0,14],'AST': latest_entry.iloc[0,15],'HR': latest_entry.iloc[0,22],
    'AR': latest_entry.iloc[0,23]}]
    
    features_inference = pd.DataFrame(features_inference_list)

    encoder = LabelEncoder()
    home_team_enc = encoder.fit_transform(features_inference['HomeTeam'])
    home_encoded_mapping = dict(
        zip(encoder.classes_, encoder.transform(encoder.classes_).tolist()))
    features_inference['HomeTeam'] = home_team_enc

    encoder = LabelEncoder()
    away_team_enc = encoder.fit_transform(features_inference['AwayTeam'])
    away_team_enc_mapping = dict(
        zip(encoder.classes_, encoder.transform(encoder.classes_).tolist()))
    features_inference['AwayTeam'] = away_team_enc
    print(features_inference_list)

   

    print(features_inference)
   
    titanic_fg = fs.get_or_create_feature_group(
        name="titanic_modal_2",
        version=1,
        primary_key=["passenger_id"], 
        description="Titanic dataset")  
    #titanic_fg.insert(titanic_df, write_options={"wait_for_job" : False})

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()