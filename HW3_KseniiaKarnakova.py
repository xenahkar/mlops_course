
import io
import os
import json
import pickle
import mlflow
import logging
import pandas as pd

from typing import Any, Dict, Literal
from datetime import timedelta, datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.models import DAG, Variable
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient


_LOG = logging.getLogger()
_LOG.addHandler(logging.StreamHandler())

BUCKET = Variable.get("S3_BUCKET")
FEATURES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]

TARGET = "MedHouseVal"

DEFAULT_ARGS = {
    "owner": "Kseniia Karnakova",
    "retry" : 3,
    "retry_delay" : timedelta(minutes=1)
}

model_names = ["random_forest", "linear_regression", "decision_tree"]

models = dict(
    zip(model_names, [
        RandomForestRegressor(),
        LinearRegression(),
        DecisionTreeRegressor(),
    ]))

dag = DAG(
    dag_id = 'KseniiaKarnakova',
    schedule_interval = "0 1 * * *",
    start_date = days_ago(2),
    catchup = False,
    tags = ["mlops"],
    default_args = DEFAULT_ARGS
    )

def configure_mlflow():
    for key in [
        "MLFLOW_TRACKING_URI",
        "AWS_ENDPOINT_URL",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_DEFAULT_REGION",
    ]:
        os.environ[key] = Variable.get(key)


def init(**kwargs) -> Dict[str, Any]:

    timestamp = datetime.now().strftime('%d/%m/%y %H:%M:%S')
    _LOG.info(f"Train pipeline started at {timestamp}.")

    configure_mlflow()

    if mlflow.get_experiment_by_name('Ksenia_Karnakova'):
        experiment_id = mlflow.set_experiment('Ksenia_Karnakova').experiment_id
    else:
        experiment_id = mlflow.create_experiment('Ksenia_Karnakova')

    with mlflow.start_run(run_name="xenahkar", experiment_id=experiment_id, description="parent") as parent_run:
        parent_run_id = parent_run.info.run_id

    metrics = {
    "init_timestamp": timestamp, 
    "parent_run_id": parent_run_id,
    "experiment_id": experiment_id
    }
    return metrics


def get_data(**kwargs) -> Dict[str, Any]:
    ti = kwargs['ti']
    metrics = ti.xcom_pull(task_ids='init')
    start_time = datetime.now().strftime('%d/%m/%y %H:%M:%S')

    _LOG.info(f"Downloading data from sklearn started at {start_time}.")
    
    data = fetch_california_housing(as_frame=True)
    data = pd.concat([data["data"], pd.DataFrame(data["target"])], axis=1)
    
    end_time = datetime.now().strftime('%d/%m/%y %H:%M:%S')
    _LOG.info(f"Data downloaded at {end_time}.")

    s3_hook = S3Hook("s3_connection")
    filebuffer = io.BytesIO()
    data.to_pickle(filebuffer)
    filebuffer.seek(0)

    s3_hook.load_file_obj(
        file_obj=filebuffer,
        key=f"KseniiaKarnakova/datasets/california_housing.pkl",
        bucket_name=BUCKET,
        replace=True,
    )

    metrics['get_data_start_time'] = start_time
    metrics['get_data_end_time'] = end_time
    metrics['data_shape'] = data.shape

    return metrics


def prepare_data(**kwargs) -> Dict[str, Any]:
    ti = kwargs['ti']
    metrics = ti.xcom_pull(task_ids='get_data')

    _LOG.info("Preparing data for training.")
    start_time = datetime.now().strftime('%d/%m/%y %H:%M:%S')

    s3_hook = S3Hook("s3_connection")
    file = s3_hook.download_file(
        key=f"KseniiaKarnakova/datasets/california_housing.pkl", 
        bucket_name=BUCKET
    )
    data = pd.read_pickle(file)

    X, y = data[FEATURES], data[TARGET]

    _LOG.info("Splitting to test and train data.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    _LOG.info("Scaling data with Standard Scaler.")

    scaler = StandardScaler()
    X_train = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns, 
        index=X_train.index
    )
    X_test = pd.DataFrame(
        scaler.transform(X_test), 
        columns=X_test.columns,
        index=X_test.index
    )

    _LOG.info("Saving data to S3.")

    for name, data in zip(
        ["X_train", "X_test", "y_train", "y_test"],
        [X_train, X_test, y_train, y_test],
    ):
        filebuffer = io.BytesIO()
        pickle.dump(data, filebuffer)
        filebuffer.seek(0)
        s3_hook.load_file_obj(
            file_obj=filebuffer,
            key=f"KseniiaKarnakova/datasets/{name}.pkl",
            bucket_name=BUCKET,
            replace=True,
        )

    end_time = datetime.now().strftime('%d/%m/%y %H:%M:%S')

    _LOG.info("Data prepared for training and saved to S3.")

    metrics['start_prepare_data'] = start_time
    metrics['finish_prepare_data'] = end_time
    metrics['features_data'] = list(X.columns)
    
    return metrics


def train_model(**kwargs) -> Dict[str, Any]:
    ti = kwargs['ti']
    metrics = ti.xcom_pull(task_ids="prepare_data")
    experiment_id = metrics['experiment_id']
    parent_run_id = metrics['parent_run_id']

    start_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    _LOG.info(f"Started training at {start_time}.")
    
    model_name = kwargs["model_name"]
    model = models[model_name]

    s3_hook = S3Hook("s3_connection")
    data = {}
    file = s3_hook.download_file(
        key=f"KseniiaKarnakova/datasets/X_train.pkl",
        bucket_name=BUCKET,
    )
    data['X_train'] = pd.read_pickle(file)
    
    file = s3_hook.download_file(
        key=f"KseniiaKarnakova/datasets/X_test.pkl",
        bucket_name=BUCKET,
    )
    data['X_test'] = pd.read_pickle(file)

    file = s3_hook.download_file(
        key=f"KseniiaKarnakova/datasets/y_train.pkl",
        bucket_name=BUCKET,
    )
    data['y_train'] = pd.read_pickle(file)

    file = s3_hook.download_file(
        key=f"KseniiaKarnakova/datasets/y_test.pkl",
        bucket_name=BUCKET,
    )
    data['y_test'] = pd.read_pickle(file)

    model.fit(data['X_train'], data['y_train'])
    prediction = model.predict(data['X_test']) 

    eval_df = data['X_test'].copy()
    eval_df['target'] = data['y_test']
    
    configure_mlflow()

    with mlflow.start_run(run_name=model_name, experiment_id=experiment_id, parent_run_id=parent_run_id, nested=True) as child_run:
        signature = infer_signature(data['X_test'], prediction)
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=model_name,
            signature=signature,
        )
        mlflow.evaluate(
            model=model_info.model_uri,
            data=eval_df,
            targets="target",
            model_type="regressor",
            evaluators=["default"],
        )

    end_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    
    _LOG.info('Model trained.')

    metrics[model_name] = {'train_start_time': start_time, 'train_end_time': end_time}
    return metrics


def save_results(**kwargs) -> None:
    ti = kwargs['ti']
    metrics = ti.xcom_pull(task_ids="prepare_data")
    metrics['random_forest'] = kwargs["task_instance"].xcom_pull(task_ids="train_random_forest")["random_forest"]
    metrics['linear_regression'] = kwargs["task_instance"].xcom_pull(task_ids="train_linear_regression")["linear_regression"]
    metrics['decision_tree'] = kwargs["task_instance"].xcom_pull(task_ids="train_decision_tree")["decision_tree"]

    s3_hook = S3Hook("s3_connection")
    filebuffer = io.BytesIO()
    filebuffer.write(json.dumps(metrics).encode())
    filebuffer.seek(0)
    s3_hook.load_file_obj(
        file_obj=filebuffer,
        key=f"KseniiaKarnakova/results/metrics.json",
        bucket_name=BUCKET,
        replace=True
        )
    
    _LOG.info("Metrics successfully saved.")


task_init = PythonOperator(task_id='init', python_callable=init, dag=dag)
task_get_data = PythonOperator(task_id='get_data', python_callable=get_data, dag=dag)
task_prepare_data = PythonOperator(task_id='prepare_data', python_callable=prepare_data, dag=dag)
training_model_tasks = [
    PythonOperator(task_id="train_random_forest", python_callable=train_model, dag=dag, provide_context=True, op_kwargs={"model_name": "random_forest"}),
    PythonOperator(task_id="train_linear_regression", python_callable=train_model, dag=dag, provide_context=True, op_kwargs={"model_name": "linear_regression"}),
    PythonOperator(task_id="train_decision_tree", python_callable=train_model, dag=dag, provide_context=True, op_kwargs={"model_name": "decision_tree"})
]
task_save_results = PythonOperator(task_id='save_results', python_callable=save_results, dag=dag)

task_init >> task_get_data >> task_prepare_data >> training_model_tasks >> task_save_results
