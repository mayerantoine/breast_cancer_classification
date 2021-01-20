
# Breast Cancer classification using Azure ML

*TODO:* Write a short introduction to your project.
Predict whether the cancer is benign or malignant using Azure Machine learning service. We  compare the results of Azure automated machine learning  and Hyperparameter tuning to solve this classification problem. Then we deploy the model on an Azure Container Serivce(ACI) as a respoint endpoint and test the endpoing by consuming it using an HTTP post request and get a prediction.

<img src="capstone-diagram_adapt.png" width="460" heigth = "400"  style="float: left; margin-right: 10px;" />

## Dataset

### Overview
*TODO*: Explain about the data you are using and where you got it from.
We used the Breast Cancer Wisconsin (Diagnostic) Data Set.Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image.
Dataset was download from Kaggle : (https://www.kaggle.com/uciml/breast-cancer-wisconsin-data)
Also can be found on UCI Machine Learning Repository: (https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)

### Task
*TODO*: Explain the task you are going to be solving with this dataset and the features you will be using for it.
We approach the project as a classification task. We wanted to classiy the cancer as cancer is benign or malignant based on the provided data.
The target column is Diagnosis (M = malignant, B = benign).


### Access
*TODO*: Explain how you are accessing the data in your workspace.
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

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
