import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from model_registry import register_model, list_models


def test_register_and_list_models():
    with tempfile.TemporaryDirectory() as tmp:
        registry_path = os.path.join(tmp, "registry.json")
        entry = register_model(
            name="test_model",
            version="v1",
            artifact_path="/tmp/model.joblib",
            registry_path=registry_path,
            metrics={"acc": 0.9},
            metadata={"note": "test"},
        )
        models = list_models(registry_path)
        assert len(models) == 1
        assert models[0]["name"] == entry["name"]
