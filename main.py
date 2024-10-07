
from codes.utils import load_mlflow
import numpy as np
import pandas as pd
import pytest

stage = "Staging"

16

def test_load_model():
    model = load_mlflow(stage=stage)
    assert model

@pytest.mark.depends (on=['test_load_model'])

def test_model_input():
    model = load_mlflow(stage=stage)
    X= np.array([1,2]).reshape(-1,2)
    X = pd.DataFrame (X, columns=['x1', 'x2'])
    pred= model.predict(X) # type: ignore
    assert pred

@pytest.mark.depends (on=["test_model_input"])
def test_model_output():
    model= load_mlflow(stage=stage)
    X= np.array([1,2]).reshape(-1,2)

    X= pd.DataFrame (X, columns=['x1', 'x2'])

    pred=  model.predict(X) 

    assert pred.shape (1,1), f"{pred.shape=}"

@pytest.mark.depends (on=['test_load_model'])

def test_model_coeff():
    model = load_mlflow(stage=stage)
    assert model.coef_shape == (1,2), f"{model.coef.shape=}"#type:igmore

