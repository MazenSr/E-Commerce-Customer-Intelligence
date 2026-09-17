import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler, StandardScaler, PowerTransformer


def get_scaler(scaler_type: str):

    scaler_type = scaler_type.lower()

    if scaler_type == "standard":
        return StandardScaler()
    elif scaler_type == "robust":
        return RobustScaler()
    else:
        raise ValueError("scaler_type must be either 'standard' or 'robust'")


def build_preprocessing_pipeline(
    skewed_cols: list[str],
    powerd_transform_cols: list[str],
    bounded_cols: list[str], 
    standard_cols: list[str],
    scaler_type: str = "standard",
    ) -> ColumnTransformer:

    """
    Build a preprocessing pipeline for customer behavioral features.

    Parameters
    ----------
    skewed_cols : list[str]
        Positive, highly right-skewed features to transform using log1p
        before scaling.

    powerd_transform_cols : list[str]
        Skewed features where Log1p fails.

    bounded_cols : list[str], optional
        Bounded behavioral features, such as ratios in [0, 1].
        These are scaled without logarithmic transformation.

    standard_cols : list[str], optional
        Continuous features that do not require log transformation
        but should still be scaled.

    scaler_type : str, default="standard"
        Scaling strategy:
        - "standard": StandardScaler
        - "robust": RobustScaler

    Returns
    -------
    ColumnTransformer
        Unfitted preprocessing pipeline.
    """

    skewed_cols = skewed_cols or []
    powerd_transform_cols = powerd_transform_cols or []
    bounded_cols = bounded_cols or []
    standard_cols = standard_cols or []


    log_pipeline = Pipeline([
        ("log1p", FunctionTransformer(np.log1p, validate=False, inverse_func=np.expm1)),
        ("scaler", get_scaler(scaler_type)),
    ])

    scaler_pipeline = Pipeline([
        ("scaler", get_scaler(scaler_type))
        ])

    if scaler_type == "robust":
        powerd_pipeline = Pipeline([
            ("power", PowerTransformer(method="yeo-johnson", standardize=False)),
            ("scaler", RobustScaler())
        ])
    else:
        powerd_pipeline = Pipeline([
            ("power", PowerTransformer(method="yeo-johnson", standardize=True))
        ])


    preprocessor = ColumnTransformer(
        [
            ("skewed", log_pipeline, skewed_cols),
            ('powered', powerd_pipeline, powerd_transform_cols),
            ("bounded", scaler_pipeline, bounded_cols),
            ("standard", scaler_pipeline, standard_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )
    preprocessor.set_output(transform="pandas")

    return preprocessor