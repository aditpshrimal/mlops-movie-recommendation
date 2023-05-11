import os
from datetime import datetime, timedelta
import random
import implicit
import numpy as np
import pandas as pd
import optuna
import scipy.sparse as sparse
from sklearn.model_selection import train_test_split

from airflow import DAG
from airflow.operators.python_operator import PythonOperator

def create_user_item_matrix(model):
    return model.user_factors.dot(model.item_factors.T)

def mse(user_item_matrix, sparse_matrix):
    user_item_array = np.asarray(user_item_matrix)
    test_indices = np.asarray(sparse_matrix.nonzero()).T
    true_ratings = sparse_matrix[test_indices[:, 0], test_indices[:, 1]].A1
    predicted_ratings = user_item_array[test_indices[:, 0], test_indices[:, 1]]

    mse_sum = np.sum((true_ratings - predicted_ratings) ** 2)
    n_total = len(test_indices)
    return mse_sum / n_total

def preprocess_data(**kwargs):
    df = pd.read_csv("/Users/aditshrimal/Desktop/MSDS/Spring2/case_studies_ml/project/data/Movies_and_TV.csv", header=None)
    df = df.sample(frac=0.2, random_state=42)
    df.columns = ["asin", "reviewerId", "overallRating", "timestamp"]
    df.sort_values("timestamp", inplace=True)
    df["user_id"] = df["reviewerId"]
    df["item_id"] = df["asin"]
    df["reviewerId"] = df["reviewerId"].astype("category").cat.codes.values
    df["asin"] = df["asin"].astype("category").cat.codes.values
    kwargs['ti'].xcom_push(key='df', value=df)

def split_data(**kwargs):
    df = kwargs['ti'].xcom_pull(key='df')
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    sparse_item_user_train = sparse.csr_matrix(
        (train_df["overallRating"], (train_df["asin"], train_df["reviewerId"]))
    )
    sparse_user_item_train = sparse.csr_matrix(
        (train_df["overallRating"], (train_df["reviewerId"], train_df["asin"]))
    )

    sparse_item_user_test = sparse.csr_matrix(
        (test_df["overallRating"], (test_df["asin"], test_df["reviewerId"]))
    )
    sparse_user_item_test = sparse.csr_matrix(
        (test_df["overallRating"], (test_df["reviewerId"], test_df["asin"]))
    )
    kwargs['ti'].xcom_push(key='sparse_user_item_train', value=sparse_user_item_train)
    kwargs['ti'].xcom_push(key='sparse_user_item_test', value=sparse_user_item_test)

def train_model(**kwargs):
    sparse_user_item_train = kwargs['ti'].xcom_pull(key='sparse_user_item_train')
    sparse_user_item_test = kwargs['ti'].xcom_pull(key='sparse_user_item_test')
    trial = 1

    def objective(trial):
        factors = trial.suggest_int('factors', 10, 50)
        regularization = trial.suggest_loguniform('regularization', 1e-5, 1e-1)
        iterations = trial.suggest_int('iterations', 10, 50)
        alpha_val = trial.suggest_int('alpha_val', 10, 100)

        model = implicit.als.AlternatingLeastSquares(
            factors=factors, regularization=regularization, iterations=iterations
        )
        data_conf =        (sparse_user_item_train * alpha_val).astype("double")
        model.fit(data_conf)

        user_item_matrix = create_user_item_matrix(model)
        test_mse = mse(user_item_matrix, sparse_user_item_test)

        return test_mse

    study = optuna.create_study(direction='minimize')
    print("Created Study")
    study.optimize(objective, n_trials=trial)
    print("Optimized Study")
    best_params = study.best_params
    print("Collected best params")
    model = implicit.als.AlternatingLeastSquares(
        factors=best_params['factors'], regularization=best_params['regularization'], iterations=best_params['iterations']
    )
    data_conf = (sparse_user_item_train * best_params['alpha_val']).astype("double")
    print("Created model")
    model.fit(data_conf)
    kwargs['ti'].xcom_push(key='model', value=model)

def end(**kwargs):
    model = kwargs['ti'].xcom_pull(key='model')
    sparse_user_item_test = kwargs['ti'].xcom_pull(key='sparse_user_item_test')
    print("Fitted data on model")
    user_item_matrix = create_user_item_matrix(model)
    print("Recreated user item matrix")
    test_mse = mse(user_item_matrix, sparse_user_item_test)
    print("Created test mse")
    print("Test MSE:", test_mse)

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2023, 5, 9),
}

dag = DAG(
    'movie_recommendation_flow',
    default_args=default_args,
    description='Movie Recommendation Flow DAG',
    schedule_interval=None,
    catchup=False,
)

preprocess_data_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    provide_context=True,
    dag=dag,
)

split_data_task = PythonOperator(
    task_id='split_data',
    python_callable=split_data,
    provide_context=True,
    dag=dag,
)

train_model_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    provide_context=True,
    dag=dag,
)

end_task = PythonOperator(
    task_id='end',
    python_callable=end,
    provide_context=True,
    dag=dag,
)

preprocess_data_task >> split_data_task >> train_model_task >> end_task