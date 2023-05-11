import logging
import os
from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from xgboost import XGBClassifier

# Setup logging
logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)


class DataHandler:
    def __init__(self, file_path: str, target: str) -> None:
        self.X = None
        self.y = None
        self.file_path = file_path
        self.target = target

    def load_data(self):
        try:
            df = pd.read_csv(self.file_path)

        except FileNotFoundError as e:
            logger.error(f"File not found: {self.file_path}")
            raise e

        self.y = df[self.target]
        self.X = df.drop([self.target], axis=1)

    def split_data(self, test_size: float, val_size: float):

        total_test_size = test_size + val_size
        rel_val_size = 1 / (total_test_size / val_size)

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=total_test_size)
        X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=rel_val_size)

        return X_train, X_test, X_val, y_train, y_test, y_val


class Model:
    def __init__(self, preprocessor: ColumnTransformer) -> None:
        self.preprocessor = preprocessor
        self.pipeline = Pipeline([("preprocessor", self.preprocessor), ("classifier", XGBClassifier())])

    def train(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        y_train: pd.DataFrame,
        y_val: pd.DataFrame,
    ):
        # preprocess the validation data set as can't catagories can't be preprocessed in pipeline
        self.preprocessor.fit(X_train)
        X_val_preprocessed = self.preprocessor.transform(X_val)

        self.label_encoder = LabelEncoder()
        y_train = self.label_encoder.fit_transform(y_train)
        y_val = self.label_encoder.transform(y_val)

        eval_set = [(X_val_preprocessed, y_val)]
        self.pipeline.fit(X_train, y_train, classifier__eval_set=eval_set, classifier__early_stopping_rounds=10)

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.DataFrame,
    ):
        y_pred = self.pipeline.predict(X_test)
        y_pred = self.label_encoder.inverse_transform(y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        logger.info("Accuracy: %.2f%%" % (accuracy * 100.0))
        logger.info(classification_report(y_test, y_pred))

    def save(self, file_path: str):
        model_dir = Path(file_path)
        os.makedirs(model_dir, exist_ok=True)
        self._save_artifact(self.pipeline, "model", model_dir)
        self._save_artifact(self.label_encoder, "label_encoder", model_dir)

    def _save_artifact(self, artifact, name: str, dir: Path):
        file_path = dir / f"{name}.joblib"
        logger.info(f"Saving {name} to {file_path}")
        dump(artifact, file_path)


def train(file_path: str):

    data_handler = DataHandler(file_path, "Adopted")
    data_handler.load_data()
    X_train, X_test, X_val, y_train, y_test, y_val = data_handler.split_data(0.2, 0.2)

    # create preprocessor
    categorical_columns = data_handler.X.select_dtypes(include=["object"]).columns
    preprocessor = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns)], remainder="passthrough"
    )

    model = Model(preprocessor)
    model.train(X_train, X_val, y_train, y_val)

    model.evaluate(X_test, y_test)

    model.save("./artifacts")


if __name__ == "__main__":
    train("gs://cloud-samples-data/ai-platform-unified/datasets/tabular/petfinder-tabular-classification.csv")
