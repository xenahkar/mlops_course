import mlflow
import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
from mlflow.models import infer_signature


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

models = {
    "linear_regression": LinearRegression(),
    "decision_tree": DecisionTreeRegressor(),
    "random_forest": RandomForestRegressor()
}

experiment_name = 'Kseniia_Karnakova'


def get_data():
    data = fetch_california_housing(as_frame=True)
    data = pd.concat([data["data"], pd.DataFrame(data["target"])], axis=1)
    return data


def prepare_data(model_name: str, data: pd.DataFrame):
    X, y = data[FEATURES], data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if model_name == 'linear_regression':
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
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":

    experiment = mlflow.get_experiment_by_name(experiment_name)
    # Создадим эксперимент/подключимся к эксперименту (если такой уже есть)
    if experiment:
        exp_id = mlflow.set_experiment(experiment_name).experiment_id
    else:
        exp_id = mlflow.create_experiment(name=experiment_name)

    # Создадим parent run
    with mlflow.start_run(run_name="xenahkar", experiment_id=exp_id, description="parent") as parent_run:
        for model_name in models.keys():
            # Запустим child run
            with mlflow.start_run(run_name=model_name, experiment_id=exp_id, nested=True) as child_run:
                # Обучим модель
                data = get_data()
                X_train, X_test, y_train, y_test = prepare_data(model_name, data)
                model = models[model_name]
                model.fit(X_train, y_train)
                prediction = model.predict(X_test)

                eval_df = X_test.copy()
                eval_df["target"] = y_test

                 # Сохраним результаты обучения 
                signature = infer_signature(X_test, prediction)
                model_info = mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path=model_name,
                    signature=signature,
                    registered_model_name=model_name,
                )

                mlflow.evaluate(
                    model=model_info.model_uri,
                    data=eval_df,
                    targets="target",
                    model_type="regressor",
                    evaluators=["default"],
                )
