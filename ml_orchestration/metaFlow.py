from metaflow import FlowSpec, step
import random
import implicit
import mlflow
import numpy as np
import pandas as pd
import scipy.sparse as sparse
import pickle
import wandb
import optuna
from sklearn.model_selection import train_test_split

class RecommendationFlow(FlowSpec):

    @step
    def start(self):
        self.next(self.load_data)

    @step
    def load_data(self):
        self.df = pd.read_csv("data/Movies_and_TV.csv", header=None)
        self.df = self.df.sample(frac=0.01, random_state=42)
        self.df.columns = ["asin", "reviewerId", "overallRating", "timestamp"]

        self.next(self.preprocess_data)

    @step
    def preprocess_data(self):
        self.df.sort_values("timestamp", inplace=True)

        self.df["user_id"] = self.df["reviewerId"]
        self.df["item_id"] = self.df["asin"]

        self.df["reviewerId"] = self.df["reviewerId"].astype("category").cat.codes.values
        self.df["asin"] = self.df["asin"].astype("category").cat.codes.values

        self.next(self.train_test_split_data)

    @step
    def train_test_split_data(self):
        self.train_df, self.test_df = train_test_split(self.df, test_size=0.2, random_state=42)

        self.sparse_item_user_train = sparse.csr_matrix(
            (self.train_df["overallRating"], (self.train_df["asin"], self.train_df["reviewerId"]))
        )
        self.sparse_user_item_train = sparse.csr_matrix(
            (self.train_df["overallRating"], (self.train_df["reviewerId"], self.train_df["asin"]))
        )

        self.sparse_item_user_test = sparse.csr_matrix(
            (self.test_df["overallRating"], (self.test_df["asin"], self.test_df["reviewerId"]))
        )
        self.sparse_user_item_test = sparse.csr_matrix(
            (self.test_df["overallRating"], (self.test_df["reviewerId"], self.test_df["asin"]))
        )

        self.next(self.optimize_hyperparameters)

    @step
    def optimize_hyperparameters(self):
        def objective(trial):
            factors = trial.suggest_int('factors', 10, 50)
            regularization = trial.suggest_loguniform('regularization', 1e-5, 1e-1)
            iterations = trial.suggest_int('iterations', 10, 50)
            alpha_val = trial.suggest_int('alpha_val', 10, 100)

            model = implicit.als.AlternatingLeastSquares(
                factors=factors, regularization=regularization, iterations=iterations, num_threads=1
            )
            data_conf = (self.sparse_user_item_train * alpha_val).astype("double")
            model.fit(data_conf)

            user_item_matrix = self.create_user_item_matrix(model)
            test_mse = self.mse(user_item_matrix, self.sparse_user_item_test)

            return test_mse

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=1)

        self.best_params = study.best_params

        self.next(self.train_final_model)

    @step
    def train_final_model(self):
        print("Starting train_final_model step...")
        self.sparse_item_user = sparse.csr_matrix(
            (self.df["overallRating"], (self.df["asin"],self.df["reviewerId"]))
        )
        self.sparse_user_item = sparse.csr_matrix(
            (self.df["overallRating"], (self.df["reviewerId"], self.df["asin"]))
        )

        factors = self.best_params['factors']
        regularization = self.best_params['regularization']
        iterations = self.best_params['iterations']
        alpha_val = self.best_params['alpha_val']

        self.model = implicit.als.AlternatingLeastSquares(
            factors=factors, regularization=regularization, iterations=iterations, num_threads=1
        )
        data_conf = (self.sparse_user_item * alpha_val).astype("double")
        self.model.fit(data_conf)
        print("Model fitting completed.")
        self.user_item_matrix = self.create_user_item_matrix(self.model)
        self.test_mse = self.mse(self.user_item_matrix, self.sparse_user_item)
        print("MSE calculation completed.")
        self.next(self.end)

    @step
    def end(self):
        print("Test MSE:", self.test_mse)

    def create_user_item_matrix(self, model):
        return model.user_factors.dot(model.item_factors.T)

    # def mse(self, user_item_matrix, sparse_matrix):
    #     user_item_array = np.asarray(user_item_matrix)
    #     test_indices = np.asarray(sparse_matrix.nonzero()).T
    #     true_ratings = sparse_matrix[test_indices[:, 0], test_indices[:, 1]].A1
    #     predicted_ratings = user_item_array[test_indices[:, 0], test_indices[:, 1]]

    #     mse_sum = np.sum((true_ratings - predicted_ratings) ** 2)
    #     n_total = len(test_indices)
    #     return mse_sum / n_total

    def mse(self, user_item_matrix, sparse_matrix):
        user_item_array = np.asarray(user_item_matrix)
        test_indices = np.asarray(sparse_matrix.nonzero()).T

        # Ensure that indices are within bounds
        max_user_index = user_item_array.shape[0] - 1
        max_item_index = user_item_array.shape[1] - 1
        test_indices = test_indices[(test_indices[:, 0] <= max_user_index) & (test_indices[:, 1] <= max_item_index)]

        true_ratings = sparse_matrix[test_indices[:, 0], test_indices[:, 1]].A1
        predicted_ratings = user_item_array[test_indices[:, 0], test_indices[:, 1]]

        mse_sum = np.sum((true_ratings - predicted_ratings) ** 2)
        n_total = len(test_indices)
        return mse_sum / n_total


if __name__ == '__main__':
    RecommendationFlow()