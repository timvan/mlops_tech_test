import pandas as pd
import pytest
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from src.predict import predict


@pytest.fixture()
def setup_dummy_model(tmp_path):

    # Setup mock data
    df = pd.DataFrame({"feature1": [1, 2, 3], "Adopted": ["yes", "no", "yes"]})
    y = df["Adopted"]
    X = df.drop(["Adopted"], axis=1)

    # Setup mock label encoder
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Setup mock model
    model = Pipeline(steps=[("classifier", LogisticRegression())])
    model.fit(X, y)

    # Save mock data set
    data_path = tmp_path / "data.csv"
    X.to_csv(data_path, index=False)

    # Save mock model
    model_path = tmp_path / "model.joblib"
    dump(model, model_path)

    # Save mock label encoder
    label_encoder_path = tmp_path / "label_encoder.joblib"
    dump(label_encoder, label_encoder_path)

    yield (data_path, model_path, label_encoder_path)


def test_predict_loads_artifacts_and_creates_outputs(setup_dummy_model, tmp_path):

    output_path = tmp_path / "result.csv"
    assert not output_path.exists()

    (data_path, model_path, label_encoder_path) = setup_dummy_model

    predict(data_path, model_path, label_encoder_path, output_path)

    assert output_path.exists()

    results = pd.read_csv(output_path)
    assert "Adopted" in results.columns
