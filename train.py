import warnings

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score
from sklearn.model_selection import train_test_split
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def metrics(actual, pred):
    accuracy = accuracy_score(actual,pred)
    f1Score = f1_score(actual,pred)
    precision = precision_score(actual,pred)
    recall = recall_score(actual,pred)
    return accuracy,f1Score,precision,recall


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file from the URL
    csv_url = (
        "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
    )
    try:
        data = pd.read_csv(csv_url, sep=",")
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e
        )


    train, test = train_test_split(data)


    train_x = train.drop(["Outcome"], axis=1)
    test_x = test.drop(["Outcome"], axis=1)
    train_y = train[["Outcome"]]
    test_y = test[["Outcome"]]

    # mlflow.start_run komutu ile modelin calistirilmasi saglanir.
              
    with mlflow.start_run(run_name='CART'):
        dtree = DecisionTreeClassifier()
                
        dtree.fit(train_x, train_y)
        
        predicted = dtree.predict(test_x)
        
        (accuracy,f1Score,precision,recall) = metrics(test_y, predicted)
        
        print("------ CART model ------")

        print("  Accuracy: %s" % accuracy)
        print("  F1 Score: %s" % f1Score)
        print("  Precision: %s" % precision)
        print("  Recall: %s" % recall)
        
        # mlflow.log_metric ile metriklerin kaydedilerek mlflow'da gosterimi saglanir.

        mlflow.log_metric("Accuracy", accuracy)
        mlflow.log_metric("F1 Score", f1Score)
        mlflow.log_metric("Precision", precision)
        mlflow.log_metric("Recall", recall)

        
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        if tracking_url_type_store != "file":
        
            # mlflow.sklearn.log_model ile modellerin kayit edilmesi saglanir.
        
            mlflow.sklearn.log_model(dtree, "model", registered_model_name="CART")
        else:
            mlflow.sklearn.log_model(dtree, "model")
            
    with mlflow.start_run(run_name='Random Forest'):
        
        randomforest = RandomForestClassifier()
                
        randomforest.fit(train_x, train_y)
        
        predicted = randomforest.predict(test_x)
        
        (accuracy,f1Score,precision,recall) = metrics(test_y, predicted)
        
        print("------ Random Forest model ------")

        print("  Accuracy: %s" % accuracy)
        print("  F1 Score: %s" % f1Score)
        print("  Precision: %s" % precision)
        print("  Recall: %s" % recall)
        

        mlflow.log_metric("Accuracy", accuracy)
        mlflow.log_metric("F1 Score", f1Score)
        mlflow.log_metric("Precision", precision)
        mlflow.log_metric("Recall", recall)

        
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        if tracking_url_type_store != "file":
        
        
            mlflow.sklearn.log_model(randomforest, "model", registered_model_name="RandomForest")
        else:
            mlflow.sklearn.log_model(randomforest, "model")
            
    with mlflow.start_run(run_name='SVM'):
        
        svmModel = svm.SVC()
                
        svmModel.fit(train_x, train_y)
        
        predicted = svmModel.predict(test_x)
        
        (accuracy,f1Score,precision,recall) = metrics(test_y, predicted)
        
        print("------ SVM ------")

        print("  Accuracy: %s" % accuracy)
        print("  F1 Score: %s" % f1Score)
        print("  Precision: %s" % precision)
        print("  Recall: %s" % recall)
        

        mlflow.log_metric("Accuracy", accuracy)
        mlflow.log_metric("F1 Score", f1Score)
        mlflow.log_metric("Precision", precision)
        mlflow.log_metric("Recall", recall)

        
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        if tracking_url_type_store != "file":
        
        
            mlflow.sklearn.log_model(svmModel, "model", registered_model_name="SVM")
        else:
            mlflow.sklearn.log_model(svmModel, "model")
            
    with mlflow.start_run(run_name='Neural Network'):
        
        NeuralNetwork = MLPClassifier()
                
        NeuralNetwork.fit(train_x, train_y)
        
        predicted = NeuralNetwork.predict(test_x)
        
        (accuracy,f1Score,precision,recall) = metrics(test_y, predicted)
        
        print("------ Neural Network ------")

        print("  Accuracy: %s" % accuracy)
        print("  F1 Score: %s" % f1Score)
        print("  Precision: %s" % precision)
        print("  Recall: %s" % recall)
        

        mlflow.log_metric("Accuracy", accuracy)
        mlflow.log_metric("F1 Score", f1Score)
        mlflow.log_metric("Precision", precision)
        mlflow.log_metric("Recall", recall)

        
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        if tracking_url_type_store != "file":
        
        
            mlflow.sklearn.log_model(NeuralNetwork, "model", registered_model_name="Neural Network")
        else:
            mlflow.sklearn.log_model(NeuralNetwork, "model")
