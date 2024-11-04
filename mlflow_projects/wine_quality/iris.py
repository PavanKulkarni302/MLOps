#Import libraries
import argparse
import warnings
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score,recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
import logging

# Set up logging
logging.basicConfig(level = logging.WARN)
logger = logging.getLogger(__name__)

# get arguments from command
parser = argparse.ArgumentParser()
parser.add_argument("--random_state", type=int, default=42)
parser.add_argument("--l1_ratio", type=float, default=0.7)
args = parser.parse_args()

#evaluation function
def eval_metrics (y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred,average="micro")
    recall = recall_score(y_test, y_pred,average="micro")
    f1 = f1_score(y_test, y_pred,average="micro")
    return accuracy, precision, recall, f1

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

#read the iris data
    df = pd.read_csv("E:/mlflow/mlflow_projects/iris_classification/iris.csv")
    df.to_csv("data/iris_dataset.csv")

#split the data into train and test
    train, test = train_test_split(df,test_size=0.2)
    X_train = train.drop("variety", axis=1)
    y_train = train["variety"]
    X_test = test.drop("variety",axis=1)
    y_test = test["variety"]
#storing train data and test data
    train.to_csv("data/train_data.csv", index=False)
    test.to_csv("data/test_data.csv",index=False)

#fit the model
    #extract parameters from argumenys
    random_state = args.random_state
    l1_ratio = args.l1_ratio
    model = LogisticRegression(random_state=random_state,l1_ratio=l1_ratio)
    model.fit(X_train,y_train)
#do prediction on test data
    y_pred = model.predict(X_test)
#evaluate the model
    accuracy, precision, recall, f1 = eval_metrics(y_test, y_pred)
    print("accuracy :",accuracy)
    print("precision:",precision)
    print("recall   :",recall)
    print("f1   :",f1)

#*************************************************************************************************************
#begining of mlflow steps
#1) set the default tracking uri
    mlflow.set_tracking_uri(uri="")
#2) set experiment
    exp = mlflow.set_experiment(experiment_name="variaty_classifications")
#3) start the run
    mlflow.start_run(experiment_id=exp.experiment_id)

#log atrifacts -----------------------------------step_1---------------------------------------------------
    input_example = train.iloc[0:5]
    mlflow.sklearn.log_model(model, "mymodel")
#log metrics -------------------------------------step_2---------------------------------------------------
#method 1
    #log single metric
    # mlflow.log_metric("accuracy",accuracy)
    # mlflow.log_metric("precision",precision)
    # mlflow.log_metric("recall",recall)
    # mlflow.log_metric("f1",f1)
# method 2
    # log multiple metric
    metrics = {
               "accuracy": accuracy,
               "precision": precision,
               "recall": recall,
               "f1": f1
              }
    mlflow.log_metrics(metrics)
#log parameters-------------------------------------step-3---------------------------------------------------
#method 1
    # #log single metris
    # mlflow.log_param("random_state",random_state)
    # mlflow.log_param("l1_ratio",l1_ratio)
# method 2
    #log multiple parmeters
    params = {
              "random_state":random_state,
              "l1_ratio":l1_ratio
              }
    mlflow.log_params(params)

#get active run detail
    run=mlflow.active_run()
    #get details from run object
    print("The experiment id is   : {}".format(run.info.experiment_id))
    print("The active run id is   : {}".format(run.info.run_id))
    print("The active run name is : {}".format(run.info.run_name))

#end the run------if not using with clause for runwith mlflow.start_run(experiment_id=exp.experiment_id)-------
    mlflow.end_run()






