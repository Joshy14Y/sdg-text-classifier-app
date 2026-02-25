import json
import os
import pickle
from pathlib import Path


class ModelLoader:
    def __init__(self):
        self.model_path = Path(os.getcwd()) / "model" / "model.pkl"
        self.labels_path = Path(os.getcwd()) / "artifacts" / "labels.json"
        self.model = self._load_model()
        self.id2label = self._load_labels()

    def _load_model(self):
        with open(self.model_path, "rb") as f:
            return pickle.load(f)

    def _load_labels(self):
        with open(self.labels_path, "rb") as f:
            return json.load(f)

    def predict(self, text: str) -> str:
        prediction = self.model.predict([text])[0]
        return self.id2label.get(str(prediction), f"ODS {prediction}")


if __name__ == "__main__":
    loader = ModelLoader()
    test_text = "acceso agua potable derecho fundamental desarrollo sostenible"
    result = loader.predict(test_text)
    print(f"Input: {test_text}")
    print(f"Predicted SDG: {result}")
