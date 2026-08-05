from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)


def get_feature_types(X):

    numeric_features = X.select_dtypes(
        include=[
            "int64",
            "float64"
        ]
    ).columns.tolist()


    categorical_features = X.select_dtypes(
        include="object"
    ).columns.tolist()


    return numeric_features, categorical_features



def build_preprocessor(
    X,
    model_type="logistic"
):

    numeric_features, categorical_features = get_feature_types(X)


    if model_type == "logistic":

        preprocessor = ColumnTransformer(
            transformers=[

                (
                    "num",
                    StandardScaler(),
                    numeric_features
                ),

                (
                    "cat",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    ),
                    categorical_features
                )

            ]
        )


    elif model_type == "tree":

        preprocessor = ColumnTransformer(
            transformers=[

                (
                    "cat",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    ),
                    categorical_features
                )

            ],

            remainder="passthrough"
        )


    return preprocessor