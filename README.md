
# Breast Cancer classification using Azure ML

Predict whether the cancer is benign or malignant using Azure Machine learning service. We  compare the results of Azure automated machine learning  and Hyperparameter tuning to solve this classification problem. Then we deploy the model on an Azure Container Serivce(ACI) as a respoint endpoint and test the endpoing by consuming it using an HTTP post request and get a prediction.
<p align='center'>
    <img src="capstone-diagram_adapt.png" width="460" heigth = "400"  style="float: left; margin-right: 10px;" />
</p>

## Dataset

### Overview

We used the Breast Cancer Wisconsin (Diagnostic) Data Set.Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image.
Dataset was download from Kaggle : (https://www.kaggle.com/uciml/breast-cancer-wisconsin-data)
Also can be found on UCI Machine Learning Repository: (https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)

### Task
We approach the project as a classification task. We wanted to classiy the cancer as cancer is benign or malignant based on the provided data.
The target column is Diagnosis (M = malignant, B = benign).


### Access
To access the data in my workspace i had to import the the csv file as a dataframe, do some cleaning, save the the as a csv before uploading in the datastore for it to be accessible via as a Tabular Dataset.

```
ds_tr = ws.get_default_datastore()
ds = Dataset.Tabular.from_delimited_files(path=ds_tr.path('cancerdata2/cancer_train_data.csv'))
```

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment
For the AutoML experiment i have the following setting.

```
automl_settings = {
    "name": "udacity_AutoML_BreastCancer_{0}".format(time.time()),
    'iterations':20,
    "experiment_timeout_minutes" : 20,
    "enable_early_stopping" : True,
    "iteration_timeout_minutes": 10,
    "n_cross_validations": 5,
    "primary_metric": 'AUC_weighted',
    "max_concurrent_iterations": 3,
    'featurization': 'auto'
}

automl_config = AutoMLConfig(task='classification',
                             compute_target=gpu_cluster,
                             training_data = training_data,
                             label_column_name = 'diagnosis',
                             run_configuration = conda_run_config,
                             **automl_settings,
                             )
```

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?
The best model from Auto ML is a VotingEnsemble with an AUC Weighted score of **0.99675** and **0.9753** accuracy.

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.
<p align='center'>
<img src="screens\best_automl_model.JPG" width="460" heigth = "400"  style="float: center; margin-right: 20px;" />
</p>

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search.
For this experiment we used a Random Forest  ,which we fine tuned using Grid Sampling for the follwing hyperparameter : max_dept, max_features,min_sample_leaf,n_estimators.

```
ps = GridParameterSampling({
    "--max_depth":choice(3,6,12,20),
    "--max_features":choice('auto','sqrt','log2'),
    "--min_samples_leaf":choice(1,3,5),
    "--n_estimator":choice(20,40,100,1000)
})


# Specify a Policy
policy = BanditPolicy(slack_factor = 0.1, evaluation_interval=1, delay_evaluation=5)

estimator = ScriptRunConfig(source_directory='./scripts',
                      script='train.py',
                      compute_target=gpu_cluster,
                      environment=keras_env)

# Create a HyperDriveConfig using the estimator, hyperparameter sampler, and policy.
hyperdrive_config = HyperDriveConfig(run_config= estimator,
                             hyperparameter_sampling=ps,
                             policy=policy,
                             primary_metric_name="accuracy",
                             primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
                             max_total_runs=50,
                             max_concurrent_runs=4,
                             max_duration_minutes= 20)
```

### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.
<p align='center'>
    <img src="screens\model_enpoint1.JPG" width="460" heigth = "400"  style="float: center; margin-right: 20px;" />
</P>

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
