import warnings
import argparse
import logging
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet, Ridge, Lasso
import mlflow
import mlflow.sklearn
from pathlib import Path
import os

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# get arguments from command
parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, required=False, default=0.7)
parser.add_argument("--l1_ratio", type=float, required=False, default=0.7)
args = parser.parse_args()

# evaluation function
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file from the URL
    data = pd.read_csv("red-wine-quality.csv")
    #os.mkdir("data/")
    data.to_csv("data/red-wine-quality.csv", index=False)
    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data, test_size=0.2)
    print("train_shape",train.shape)
    print("test_shape",test.shape)

    # train.to_csv("data/train.csv",index=False)
    # test.to_csv("data/test.csv",index=False)
    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    alpha = args.alpha
    l1_ratio = args.l1_ratio

    mlflow.set_tracking_uri(uri="")

    #print("The set tracking uri is ", mlflow.get_tracking_uri())

######################################################### First Experimnt ElasticNet #################################
    print("First Experimnt ElasticNet ")
    exp = mlflow.set_experiment(experiment_name="exp_multi_ElasticNet")
    #get_exp = mlflow.get_experiment(exp_id)

    print("Experiment Name: {} ".format(exp.name))
    print("Experiment id  : {} ".format(exp.experiment_id))
    #print("Artifact Location: {}".format(exp.artifact_location))
    #print("Tags: {}".format(exp.tags))
    #print("Lifecycle_stage: {}".format(exp.lifecycle_stage))
    #print("Creation timestamp: {}".format(exp.creation_time))

#####################################################--Run-1#############################################################
    mlflow.start_run(run_name="run-1.1")
    #get the active run details
    run = mlflow.active_run()
    print("Active run id is {}".format(run.info.run_id))
    print("Active run name is {}".format(run.info.run_name))
#log tags
    tags = {
        "engineering": "ML platform",
        "release.candidate":"RC1",
        "release.version": "2.0"
    }

    mlflow.set_tags(tags)
#fit model and prediction
    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    lr.fit(train_x, train_y)

    predicted_qualities = lr.predict(test_x)
    #model eval
    (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

    print("Elasticnet model (alpha={:f}, l1_ratio={:f}):".format(alpha, l1_ratio))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)

#log parameters
    params ={
        "alpha": alpha,
        "l1_ratio": l1_ratio
    }
    mlflow.log_params(params)

#log metrics
    metrics = {
        "rmse":rmse,
        "r2":r2,
        "mae":mae
    }
    mlflow.log_metrics(metrics)

#log model
    # Example input for signature inference
    input_example = train_x.iloc[:5]  # Use a few rows as input example
    mlflow.sklearn.log_model(lr, "mymodel",input_example = input_example)
    mlflow.log_artifacts("data/")
    #get artifact path
    artifacts_uri=mlflow.get_artifact_uri()
    print("The artifact path is",artifacts_uri )

    mlflow.end_run()

    #####################################################--Run-2#############################################################
    mlflow.start_run(run_name="run-2.1")
    # get the active run details
    run = mlflow.active_run()
    print("Active run id is {}".format(run.info.run_id))
    print("Active run name is {}".format(run.info.run_name))
    # log tags
    tags = {
        "engineering": "ML platform",
        "release.candidate": "RC1",
        "release.version": "2.0"
    }

    mlflow.set_tags(tags)

# fit model and prediction
    # hardcoding hyperparameters for experiment
    alpha = 0.5
    l1_ratio = 0.5

    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    lr.fit(train_x, train_y)

    predicted_qualities = lr.predict(test_x)
    # model eval
    (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

    print("Elasticnet model (alpha={:f}, l1_ratio={:f}):".format(alpha, l1_ratio))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)

    # log parameters
    params = {
        "alpha": alpha,
        "l1_ratio": l1_ratio
    }
    mlflow.log_params(params)

    # log metrics
    metrics = {
        "rmse": rmse,
        "r2": r2,
        "mae": mae
    }
    mlflow.log_metrics(metrics)

    # log model
    # Example input for signature inference
    input_example = train_x.iloc[:5]  # Use a few rows as input example
    mlflow.sklearn.log_model(lr, "mymodel", input_example=input_example)
    mlflow.log_artifacts("data/")
    # get artifact path
    artifacts_uri = mlflow.get_artifact_uri()
    print("The artifact path is", artifacts_uri)

    mlflow.end_run()

    #####################################################--Run-3#############################################################
    mlflow.start_run(run_name="run-3.1")
    # get the active run details
    run = mlflow.active_run()
    print("Active run id is {}".format(run.info.run_id))
    print("Active run name is {}".format(run.info.run_name))
    # log tags
    tags = {
        "engineering": "ML platform",
        "release.candidate": "RC1",
        "release.version": "2.0"
    }

    mlflow.set_tags(tags)

# fit model and prediction
    # hardcoding hyperparameters for experiment
    alpha = 0.4
    l1_ratio = 0.4
    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    lr.fit(train_x, train_y)

    predicted_qualities = lr.predict(test_x)
    # model eval
    (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

    print("Elasticnet model (alpha={:f}, l1_ratio={:f}):".format(alpha, l1_ratio))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)

    # log parameters
    params = {
        "alpha": alpha,
        "l1_ratio": l1_ratio
    }
    mlflow.log_params(params)

    # log metrics
    metrics = {
        "rmse": rmse,
        "r2": r2,
        "mae": mae
    }
    mlflow.log_metrics(metrics)

    # log model
    # Example input for signature inference
    input_example = train_x.iloc[:5]  # Use a few rows as input example
    mlflow.sklearn.log_model(lr, "mymodel", input_example=input_example)
    mlflow.log_artifacts("data/")
    # get artifact path
    artifacts_uri = mlflow.get_artifact_uri()
    print("The artifact path is", artifacts_uri)

    mlflow.end_run()

##########################################################Second Experimnt Ridge #################################
    print("Second Experimnt Ridge ")
    exp = mlflow.set_experiment(experiment_name="exp_multi_Ridge")
    #get_exp = mlflow.get_experiment(exp_id)

    print("Experiment Name: {} ".format(exp.name))
    print("Experiment id  : {} ".format(exp.experiment_id))
    #print("Artifact Location: {}".format(exp.artifact_location))
    #print("Tags: {}".format(exp.tags))
    #print("Lifecycle_stage: {}".format(exp.lifecycle_stage))
    #print("Creation timestamp: {}".format(exp.creation_time))

#####################################################--Run-1#############################################################
    mlflow.start_run(run_name="run-1.1")
    #get the active run details
    run = mlflow.active_run()
    print("Active run id is {}".format(run.info.run_id))
    print("Active run name is {}".format(run.info.run_name))
#log tags
    tags = {
        "engineering": "ML platform",
        "release.candidate":"RC1",
        "release.version": "2.0"
    }

    mlflow.set_tags(tags)
#fit model and prediction
    lr = ElasticNet(alpha=alpha, random_state=42)
    lr.fit(train_x, train_y)

    predicted_qualities = lr.predict(test_x)
    #model eval
    (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

    print("Ridge model (alpha={:f}):".format(alpha))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)

#log parameters
    params ={
        "alpha": alpha
    }
    mlflow.log_params(params)

#log metrics
    metrics = {
        "rmse":rmse,
        "r2":r2,
        "mae":mae
    }
    mlflow.log_metrics(metrics)

#log model
    # Example input for signature inference
    input_example = train_x.iloc[:5]  # Use a few rows as input example
    mlflow.sklearn.log_model(lr, "mymodel",input_example = input_example)
    mlflow.log_artifacts("data/")
    #get artifact path
    artifacts_uri=mlflow.get_artifact_uri()
    print("The artifact path is",artifacts_uri )

    mlflow.end_run()

    #####################################################--Run-2#############################################################
    mlflow.start_run(run_name="run-2.1")
    # get the active run details
    run = mlflow.active_run()
    print("Active run id is {}".format(run.info.run_id))
    print("Active run name is {}".format(run.info.run_name))
    # log tags
    tags = {
        "engineering": "ML platform",
        "release.candidate": "RC1",
        "release.version": "2.0"
    }

    mlflow.set_tags(tags)

# fit model and prediction
    # hardcoding hyperparameters for experiment
    alpha = 0.5
    l1_ratio = 0.5

    lr = ElasticNet(alpha=alpha, random_state=42)
    lr.fit(train_x, train_y)

    predicted_qualities = lr.predict(test_x)
    # model eval
    (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

    print("Ridge model (alpha={:f}):".format(alpha))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)

    # log parameters
    params = {
        "alpha": alpha
    }
    mlflow.log_params(params)

    # log metrics
    metrics = {
        "rmse": rmse,
        "r2": r2,
        "mae": mae
    }
    mlflow.log_metrics(metrics)

    # log model
    # Example input for signature inference
    input_example = train_x.iloc[:5]  # Use a few rows as input example
    mlflow.sklearn.log_model(lr, "mymodel", input_example=input_example)
    mlflow.log_artifacts("data/")
    # get artifact path
    artifacts_uri = mlflow.get_artifact_uri()
    print("The artifact path is", artifacts_uri)

    mlflow.end_run()

    #####################################################--Run-3#############################################################
    mlflow.start_run(run_name="run-3.1")
    # get the active run details
    run = mlflow.active_run()
    print("Active run id is {}".format(run.info.run_id))
    print("Active run name is {}".format(run.info.run_name))
    # log tags
    tags = {
        "engineering": "ML platform",
        "release.candidate": "RC1",
        "release.version": "2.0"
    }

    mlflow.set_tags(tags)

# fit model and prediction
    # hardcoding hyperparameters for experiment
    alpha = 0.4
    l1_ratio = 0.4
    lr = ElasticNet(alpha=alpha, random_state=42)
    lr.fit(train_x, train_y)

    predicted_qualities = lr.predict(test_x)
    # model eval
    (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

    print("Ridge model (alpha={:f}):".format(alpha))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)

    # log parameters
    params = {
        "alpha": alpha
    }
    mlflow.log_params(params)

    # log metrics
    metrics = {
        "rmse": rmse,
        "r2": r2,
        "mae": mae
    }
    mlflow.log_metrics(metrics)

    # log model
    # Example input for signature inference
    input_example = train_x.iloc[:5]  # Use a few rows as input example
    mlflow.sklearn.log_model(lr, "mymodel", input_example=input_example)
    mlflow.log_artifacts("data/")
    # get artifact path
    artifacts_uri = mlflow.get_artifact_uri()
    print("The artifact path is", artifacts_uri)

    mlflow.end_run()


######################################################### Third Experimnt Lasso #################################
    print("Third Experimnt Lasso ")
    exp = mlflow.set_experiment(experiment_name="exp_multi_Lasso")
    #get_exp = mlflow.get_experiment(exp_id)

    print("Experiment Name: {} ".format(exp.name))
    print("Experiment id  : {} ".format(exp.experiment_id))
    #print("Artifact Location: {}".format(exp.artifact_location))
    #print("Tags: {}".format(exp.tags))
    #print("Lifecycle_stage: {}".format(exp.lifecycle_stage))
    #print("Creation timestamp: {}".format(exp.creation_time))

#####################################################--Run-1#############################################################
    mlflow.start_run(run_name="run-1.1")
    #get the active run details
    run = mlflow.active_run()
    print("Active run id is {}".format(run.info.run_id))
    print("Active run name is {}".format(run.info.run_name))
#log tags
    tags = {
        "engineering": "ML platform",
        "release.candidate":"RC1",
        "release.version": "2.0"
    }

    mlflow.set_tags(tags)
#fit model and prediction
    lr = ElasticNet(alpha=alpha, random_state=42)
    lr.fit(train_x, train_y)

    predicted_qualities = lr.predict(test_x)
    #model eval
    (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

    print("Lasso model (alpha={:f}):".format(alpha))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)

#log parameters
    params ={
        "alpha": alpha
    }
    mlflow.log_params(params)

#log metrics
    metrics = {
        "rmse":rmse,
        "r2":r2,
        "mae":mae
    }
    mlflow.log_metrics(metrics)

#log model
    # Example input for signature inference
    input_example = train_x.iloc[:5]  # Use a few rows as input example
    mlflow.sklearn.log_model(lr, "mymodel",input_example = input_example)
    mlflow.log_artifacts("data/")
    #get artifact path
    artifacts_uri=mlflow.get_artifact_uri()
    print("The artifact path is",artifacts_uri )

    mlflow.end_run()

    #####################################################--Run-2#############################################################
    mlflow.start_run(run_name="run-2.1")
    # get the active run details
    run = mlflow.active_run()
    print("Active run id is {}".format(run.info.run_id))
    print("Active run name is {}".format(run.info.run_name))
    # log tags
    tags = {
        "engineering": "ML platform",
        "release.candidate": "RC1",
        "release.version": "2.0"
    }

    mlflow.set_tags(tags)

# fit model and prediction
    # hardcoding hyperparameters for experiment
    alpha = 0.5
    l1_ratio = 0.5

    lr = ElasticNet(alpha=alpha, random_state=42)
    lr.fit(train_x, train_y)

    predicted_qualities = lr.predict(test_x)
    # model eval
    (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

    print("Lasso model (alpha={:f}):".format(alpha))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)

    # log parameters
    params = {
        "alpha": alpha
    }
    mlflow.log_params(params)

    # log metrics
    metrics = {
        "rmse": rmse,
        "r2": r2,
        "mae": mae
    }
    mlflow.log_metrics(metrics)

    # log model
    # Example input for signature inference
    input_example = train_x.iloc[:5]  # Use a few rows as input example
    mlflow.sklearn.log_model(lr, "mymodel", input_example=input_example)
    mlflow.log_artifacts("data/")
    # get artifact path
    artifacts_uri = mlflow.get_artifact_uri()
    print("The artifact path is", artifacts_uri)

    mlflow.end_run()

    #####################################################--Run-3#############################################################
    mlflow.start_run(run_name="run-3.1")
    # get the active run details
    run = mlflow.active_run()
    print("Active run id is {}".format(run.info.run_id))
    print("Active run name is {}".format(run.info.run_name))
    # log tags
    tags = {
        "engineering": "ML platform",
        "release.candidate": "RC1",
        "release.version": "2.0"
    }

    mlflow.set_tags(tags)

# fit model and prediction
    # hardcoding hyperparameters for experiment
    alpha = 0.4
    l1_ratio = 0.4
    lr = ElasticNet(alpha=alpha, random_state=42)
    lr.fit(train_x, train_y)

    predicted_qualities = lr.predict(test_x)
    # model eval
    (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

    print("Lasso model (alpha={:f}):".format(alpha))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)

    # log parameters
    params = {
        "alpha": alpha
    }
    mlflow.log_params(params)

    # log metrics
    metrics = {
        "rmse": rmse,
        "r2": r2,
        "mae": mae
    }
    mlflow.log_metrics(metrics)

    # log model
    # Example input for signature inference
    input_example = train_x.iloc[:5]  # Use a few rows as input example
    mlflow.sklearn.log_model(lr, "mymodel", input_example=input_example)
    mlflow.log_artifacts("data/")
    # get artifact path
    artifacts_uri = mlflow.get_artifact_uri()
    print("The artifact path is", artifacts_uri)

    mlflow.end_run()

    #get recent run details
    recent_run = mlflow.last_active_run()
    print("The Recent Run id {}: ".format(recent_run.info.run_id))
    print("The Recent Run Name {}: ".format(recent_run.info.run_name))

