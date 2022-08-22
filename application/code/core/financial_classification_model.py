from typing import Any, List, Tuple

from category_encoders import CountEncoder
from mlflow.pyfunc import PythonModel
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder

from application.code.core.feature_engineering import engineer_features
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

    def predict(self, context: Any, content: DataFrame) -> List[Tuple[str, int]]:

        # fmt: off
        clean_content = (
            content
            .pipe(clean_data, self.categorical_columns)
            .pipe(engineer_features)
        )
        # fmt: on

        X = generate_features(
            clean_content,
            columns_selection=self.columns_selection,
            binary_columns=self.binary_columns,
            categorical_encoder=self.categorical_encoder,
        )

        preds = self.model.predict(X)
        label_preds = self.label_encoder.inverse_transform(preds)

        return list(zip(label_preds, preds))
