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


def train(file_path: str):

    # load data
    df = pd.read_csv(file_path)

    # split data
    target = "Adopted"

    y = df[target]
    X = df.drop([target], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5)

    # create preprocessor
    categorical_columns = X.select_dtypes(include=["object"]).columns
    preprocessor = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns)], remainder="passthrough"
    )

    preprocessor.fit(X_train)
    X_val_preprocessed = preprocessor.transform(X_val)

    # label encode the target variable
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_val = label_encoder.transform(y_val)

    # setup pipeline
    pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", XGBClassifier())])

    # train model
    eval_set = [(X_val_preprocessed, y_val)]
    pipeline.fit(X_train, y_train, classifier__eval_set=eval_set, classifier__early_stopping_rounds=10)

    # evaluate model
    y_pred = pipeline.predict(X_test)
    y_pred = label_encoder.inverse_transform(y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print(classification_report(y_test, y_pred))

    # save model
    model_dir = Path("./artifacts")
    os.makedirs(model_dir, exist_ok=True)
    dump(pipeline, model_dir / "model")
    dump(label_encoder, model_dir / "label_encoder")


if __name__ == "__main__":
    train("gs://cloud-samples-data/ai-platform-unified/datasets/tabular/petfinder-tabular-classification.csv")
