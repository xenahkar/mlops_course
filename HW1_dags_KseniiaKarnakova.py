import io
import json
import logging
import numpy as np
import os
import pandas as pd
import pickle
from datetime import datetime, timedelta
from airflow.models import DAG, Variable
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, median_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
from airflow.utils.dates import days_ago
from airflow.operators.python_operator import PythonOperator


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
    "retries": 3,
    "retry_delay": timedelta(minutes=1)
}

models = {
    "linear_regression": LinearRegression(),
    "decision_tree": DecisionTreeRegressor(),
    "random_forest": RandomForestRegressor()
}


def init(model_name: str):
    timestamp = datetime.now().strftime('%d/%m/%y %H:%M:%S')
    _LOG.info("Train pipeline started for {model_name} model.")

    init_info = {
    "timestamp": timestamp, 
    "model_name": model_name
    }
    return init_info


def get_data(**kwargs):
    ti = kwargs['ti']
    init_metadata = ti.xcom_pull(task_ids='init')

    _LOG.info("Downloading data from sklearn.")
    start_time = datetime.now().strftime('%d/%m/%y %H:%M:%S')
    
    data = fetch_california_housing(as_frame=True)
    data = pd.concat([data["data"], pd.DataFrame(data["target"])], axis=1)
    
    end_time = datetime.now().strftime('%d/%m/%y %H:%M:%S')
    _LOG.info("Data downloaded.")

    s3_hook = S3Hook("s3_connection")
    filebuffer = io.BytesIO()
    data.to_pickle(filebuffer)
    filebuffer.seek(0)

    # Сохраняю файл в формате pkl на S3
    s3_hook.load_file_obj(
        file_obj=filebuffer,
        key=f"KseniiaKarnakova/{init_metadata['model_name']}/datasets/california_housing.pkl",
        bucket_name=BUCKET,
        replace=True,
    )

    get_data_info = {
        "start_time": start_time,
        "end_time": end_time,
        "size": data.shape
    }
    
    return get_data_info


def prepare_data(**kwargs):
    ti = kwargs['ti']
    init_metadata = ti.xcom_pull(task_ids='init')

    # Загружаю готовые данные с S3
    s3_hook = S3Hook("s3_connection")
    file = s3_hook.download_file(
        key=f"KseniiaKarnakova/{init_metadata['model_name']}/datasets/california_housing.pkl", 
        bucket_name=BUCKET
    )
    data = pd.read_pickle(file)

    _LOG.info("Preparing data for training.")
    start_time = datetime.now().strftime('%d/%m/%y %H:%M:%S')

    # Делю данные на фичи и таргет
    X, y = data[FEATURES], data[TARGET]

    _LOG.info("Splitting to test and train data.")
    # Делю данные на обучение и тест
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    if init_metadata['model_name'] == 'linear_regression':
        # для линейной регрессии нормализую данные
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

    _LOG.info("Saving splitted to train and test data to S3.")

    # Сохраняю файлы в формате pkl на S3
    for name, data in zip(
        ["X_train", "X_test", "y_train", "y_test"],
        [X_train, X_test, y_train, y_test],
    ):
        filebuffer = io.BytesIO()
        pickle.dump(data, filebuffer)
        filebuffer.seek(0)
        s3_hook.load_file_obj(
            file_obj=filebuffer,
            key=f"KseniiaKarnakova/{init_metadata['model_name']}/datasets/{name}.pkl",
            bucket_name=BUCKET,
            replace=True,
        )

    end_time = datetime.now().strftime('%d/%m/%y %H:%M:%S')
    _LOG.info("Data prepared for training and saved to S3.")

    prepare_data_info = {
        "start_time": start_time,
        "end_time": end_time,
        "features": list(X.columns)
    }
    
    return prepare_data_info

def train_model(**kwargs):
    ti = kwargs['ti']
    init_metadata = ti.xcom_pull(task_ids='init')
    
    # Загружаю готовые данные с S3
    s3_hook = S3Hook("s3_connection")
    data = {}
    for name in ["X_train", "X_test", "y_train", "y_test"]:
        file = s3_hook.download_file(
            key=f"KseniiaKarnakova/{init_metadata['model_name']}/datasets/{name}.pkl",
            bucket_name=BUCKET,
        )
        data[name] = pd.read_pickle(file)

    _LOG.info(f"Training {init_metadata['model_name']} model.")
    start_time = datetime.now().strftime('%d/%m/%y %H:%M:%S')

    model = models[init_metadata['model_name']]
    model.fit(data["X_train"], data["y_train"])
    prediction = model.predict(data["X_test"])

    result = {}
    result["r2_score"] = r2_score(data["y_test"], prediction)
    result["rmse"] = mean_squared_error(data["y_test"], prediction) ** 0.5
    result["mae"] = median_absolute_error(data["y_test"], prediction)
    
    end_time = datetime.now().strftime('%d/%m/%y %H:%M:%S')
    _LOG.info(f"Finished training model.")
    
    model_metrics = {
        "start_time": start_time,
        "end_time": end_time,
        "metrics": result
    }
    
    return model_metrics

def save_results(**kwargs):
    ti = kwargs['ti']
    init_metadata = ti.xcom_pull(task_ids='init')
    model_metrics = ti.xcom_pull(task_ids='train_model')

    _LOG.info("Saving metrics to S3.")
    
    # Сохраняю метрики в формате json на S3
    s3_hook = S3Hook("s3_connection")
    filebuffer = io.BytesIO()
    filebuffer.write(json.dumps(model_metrics['metrics']).encode())
    filebuffer.seek(0)
    s3_hook.load_file_obj(
        file_obj=filebuffer,
        key=f"KseniiaKarnakova/{init_metadata['model_name']}/results/metrics.json",
        bucket_name=BUCKET,
        replace=True
    )

    _LOG.info(f"Saved metrics for {init_metadata['model_name']}: {model_metrics['metrics']}.")

def create_dag(dag_id: str, model_name: str):
    dag = DAG(
        dag_id=dag_id,
        schedule_interval="0 1 * * *",
        start_date=days_ago(1),
        default_args=DEFAULT_ARGS,
        tags=["mlops"]
    )

    with dag:
        task_init = PythonOperator(
            task_id="init",
            python_callable=init,
            op_kwargs={"model_name": model_name}
        )

        task_get_data = PythonOperator(
            task_id="get_data",
            python_callable=get_data,
        )

        task_prepare_data = PythonOperator(
            task_id="prepare_data",
            python_callable=prepare_data,
        )

        task_train_model = PythonOperator(
            task_id="train_model",
            python_callable=train_model,
        )

        task_save_results = PythonOperator(
            task_id="save_results",
            python_callable=save_results,
        )

        task_init >> task_get_data >> task_prepare_data >> task_train_model >> task_save_results

    return dag

for model_name in models.keys():
    globals()[f"KseniiaKarnakova_{model_name}"] = create_dag(f"KseniiaKarnakova_{model_name}", model_name)