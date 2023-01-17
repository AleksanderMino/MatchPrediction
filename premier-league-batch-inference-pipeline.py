import os
import modal

LOCAL = True

if LOCAL == False:
   stub = modal.Stub()
   hopsworks_image = modal.Image.debian_slim().pip_install(["hopsworks==3.0.4","joblib","seaborn","sklearn","dataframe-image"])
   @stub.function(image=hopsworks_image, schedule=modal.Period(days=8), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
   def f():
       g()

def g():
    import pandas as pd
    import hopsworks
    import joblib
    import datetime
    from PIL import Image
    from datetime import datetime
    import dataframe_image as dfi
    from sklearn.metrics import confusion_matrix
    from matplotlib import pyplot
    import seaborn as sns
    import requests

    project = hopsworks.login()
    fs = project.get_feature_store()

    mr = project.get_model_registry()
    model = mr.get_model("premier_league_model", version=3)
    model_dir = model.download()
    model = joblib.load(model_dir + "/premier_league_model.pkl")

    feature_view = fs.get_feature_view(name="premier_league", version=3)
    batch_data = feature_view.get_batch_data()

    y_pred = model.predict(batch_data)
    # print(y_pred)
    result = y_pred[y_pred.size-1]
    print('Predicted result in binary: ', result)
    if result == 1:
        print("Home Team Won The Game!!" )

    elif result == 0:
        print("The result of the game was a draw")
    else:
        print("Away Team Won the game")
        #passenger_icon = "https://raw.githubusercontent.com/AleksanderMino/Serverless-ML-Titanic/main/Dicaprio_fate.jpeg"
    #img = Image.open(requests.get(passenger_icon,stream=True).raw)
    #img.save("./latest_passenger.png")
    dataset_api = project.get_dataset_api()
    dataset_api.upload("./latest_passenger.png", "Resources/images", overwrite=True)

    premier_league_fg = fs.get_feature_group(name="premier_league", version=6)
    df = premier_league_fg.read()
    print(df["result"])
    label_binary = df.iloc[-1]["result"]
    print('actual result in binary ',label_binary)
    if label_binary==1:
        label = "Home Team Won"
    elif label_binary==0:
        label = "The result was a draw"
    else:
        label = "Away Team Won"

    if label == 'Survived':
        print("Home Team Won!!" )
        #passenger_icon = "https://raw.githubusercontent.com/AleksanderMino/Serverless-ML-Titanic/main/survived.jpeg"
    elif label == "The result was a draw":
        print("Draw!!")
    else:
        print("Away Team Won")
        #passenger_icon = "https://raw.githubusercontent.com/AleksanderMino/Serverless-ML-Titanic/main/Dicaprio_fate.jpeg"
    #img = Image.open(requests.get(passenger_icon, stream=True).raw)
    #img.save("./actual_fate.png")
    #dataset_api.upload("./actual_fate.png", "Resources/images", overwrite=True)

    monitor_fg = fs.get_or_create_feature_group(name="premier_league_predictions",
                                                version=1,
                                                primary_key=["datetime"],
                                                description="Premier League Prediction/Outcome Monitoring"
                                                )

    now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    data = {
        'prediction' : [result],
        'label' : [float(label_binary)],
        'datetime': [now] 
    }

    monitor_df = pd.DataFrame(data)
    monitor_fg.insert(monitor_df, write_options={"wait_for_job" : False})

    history_df = monitor_fg.read()
    history_df = pd.concat([history_df, monitor_df])

    df_recent = history_df.tail(5)
    dfi.export(df_recent, './df_recent.png', table_conversion = 'matplotlib')
    dataset_api.upload("./df_recent.png", "Resources/images", overwrite=True)
    
    predictions = history_df[['prediction']]
    labels = history_df[['label']]
    print(predictions)
    print(labels)

    print("Number of different fate predictions to date: " + str(predictions.value_counts().count()))
    if predictions.value_counts().count() == 2:
        results = confusion_matrix(labels, predictions)
        print(results)
        df_cm = pd.DataFrame(results, ['True Result: Home Team Won', 'True Result: Draw', 'True Result: Away Team Won'],
                             ['Predicted Result: Home Team Won', 'Predicted Result: Draw', 'Predicted Result: Away Team Won'])
        cm = sns.heatmap(df_cm, annot=True)
        fig = cm.get_figure()
        fig.savefig("./confusion_matrix.png")
        dataset_api.upload("./confusion_matrix.png", "Resources/images", overwrite=True)
    else:
        print("You need 3 different passenger's fate predictions to create the confusion matrix.")
        print("Run the batch inference pipeline more times until you get 3 different passenger's fate predictions")

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()

