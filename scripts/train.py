from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from azureml.core import Dataset
import pandas as pd


#df = df.to_pandas_dataframe().dropna()
#df = df.drop('Unnamed: 32',axis=1)
#df = df.drop('id',axis=1)

def clean_data(folder):
    
    run = Run.get_context()
    ws = run.experiment.workspace
    ds_tr = ws.get_default_datastore()
    ds = Dataset.Tabular.from_delimited_files(path=ds_tr.path('cancerdata2/cancer_train_data.csv'))
    
    
    #df = pd.read_csv(os.path.join(folder,'cancer_train_data.csv'))
    df = ds.to_pandas_dataframe()
    print(df)
    #y = df['diagnosis'].apply(lambda x: 1 if x =='B' else 0).astype('category').copy()
    y = df['diagnosis'].astype('category').copy()
    X = df.drop('diagnosis',axis=1)

    print("X cols:",X.columns)
    print("Training shape:", X.shape)

    x_train,x_test,y_train,y_test = train_test_split(X,y,random_state=42,test_size=0.25,stratify=y)
    
    return x_train,x_test,y_train,y_test 



def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_folder',type=str,default='./data')
    parser.add_argument('--n_estimators', type=int, default=40)
    parser.add_argument('--max_depth', type=int, default=100)
    parser.add_argument('--max_features',type=str,default='auto')
    parser.add_argument('--min_samples_leaf',type=int,default =3)
    
    run = Run.get_context()
    args = parser.parse_args()
    
    data_folder = args.data_folder
    n_estimators = args.n_estimators
    max_depth = args.max_depth
    max_features = args.max_features
    min_samples_leaf = args.min_samples_leaf
    
    print(data_folder)
    x_train,x_test,y_train,y_test = clean_data(data_folder)

    run.log("n_estimator:", np.int(args.n_estimators))
    run.log("max_depth:", np.int(args.max_depth))
    run.log("max_features:", args.max_features)
    run.log("min_samples_leaf:", np.int(args.min_samples_leaf))

    model = RandomForestClassifier(n_estimators = n_estimators,
                                   max_depth = max_depth,
                                   min_samples_leaf =min_samples_leaf, 
                                   max_features = max_features,
                                   random_state = 41,
                                   n_jobs=-1)
    
    model.fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("accuracy", np.float(accuracy))
    
    f1score = f1_score(model.predict(x_test),y_test)
    run.log("f1score", np.float(f1score))

    
     # Write the model to file.
    model_path = "./outputs/cancer_model.pkl"
    os.makedirs("outputs", exist_ok=True)
    print('Saving the model to {}'.format(model_path))
    joblib.dump(model, model_path)

if __name__ == '__main__':
    main()
