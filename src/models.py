from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from preprocessing import build_preprocessor

def build_logistic_model(X):

    model = Pipeline(
        steps=[

            (
                "preprocessor",
                build_preprocessor(
                    X,
                    model_type="logistic"
                )
            ),

            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42
                )
            )

        ]
    )

    return model




def build_random_forest_model(X):

    model = Pipeline(
        steps=[

            (
                "preprocessor",
                build_preprocessor(
                    X,
                    model_type="tree"
                )
            ),

            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=42,
                    n_jobs=-1
                )
            )

        ]
    )

    return model