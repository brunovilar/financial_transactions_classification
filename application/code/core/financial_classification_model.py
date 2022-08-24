from typing import Any, List, Tuple

from category_encoders import CountEncoder
from mlflow.pyfunc import PythonModel
from numpy import ndarray
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder

from application.code.core.feature_engineering import (
    engineer_features,
    standardize_label,
)
from application.code.core.model_training import (
    clean_data,
    combine_feature_columns,
    generate_features,
)


class FinancialClassificationModel(PythonModel):
    def __init__(
        self,
        categorical_columns: List[str],
        binary_columns: List[str],
        numeric_columns: List[str],
        label_encoder: LabelEncoder,
        categorical_encoder: CountEncoder,
        model: Any,
    ):
        self.categorical_columns = categorical_columns
        self.binary_columns = binary_columns
        self.numeric_columns = numeric_columns
        self.label_encoder = label_encoder
        self.categorical_encoder = categorical_encoder
        self.model = model

        self.columns_selection = combine_feature_columns(
            categorical_encoder.cols,
            categorical_columns,
            numeric_columns,
            binary_columns,
        )

    def preprocess_data(self, data: DataFrame) -> DataFrame:

        return data.pipe(clean_data, self.categorical_columns).pipe(engineer_features)

    def generate_features(self, data: DataFrame) -> ndarray:

        return generate_features(
            data,
            columns_selection=self.columns_selection,
            binary_columns=self.binary_columns,
            categorical_encoder=self.categorical_encoder,
        )

    def predict(self, data: DataFrame) -> List[str]:

        processed_data = self.preprocess_data(data)
        X = self.generate_features(processed_data)
        preds = self.model.predict(X)
        label_preds = self.label_encoder.inverse_transform(preds)

        return label_preds

    def encode_labels(self, labels: List[str]) -> List[int]:

        classes = set(self.label_encoder.classes_)
        clean_labels = (standardize_label(label) for label in labels)

        return [
            int(self.label_encoder.transform([label])) if label in classes else -1
            for label in clean_labels
        ]
