import os

import pandas as pd
from joblib import load
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder


def predict(data_path: str, model_path: str, label_encoder_path: str, output_path: str):

    pipeline: Pipeline = load(model_path)

    X = pd.read_csv(data_path)

    y_pred = pipeline.predict(X)

    label_encoder: LabelEncoder = load(label_encoder_path)
    y_pred = pd.DataFrame(label_encoder.inverse_transform(y_pred), columns=["Adopted"])

    # Not sure whether to join original feature data
    # df = pd.concat([X, y_pred], axis=1)
    df = pd.concat([y_pred], axis=1)

    # save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    predict(
        "gs://cloud-samples-data/ai-platform-unified/datasets/tabular/petfinder-tabular-classification.csv",
        "./artifacts/model.joblib",
        "./artifacts/label_encoder.joblib",
        "./output/results.csv",
    )
