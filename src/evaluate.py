import numpy as np

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve
)



def evaluate_model(
    y_true,
    prob,
    threshold=0.5
):

    pred = (
        prob >= threshold
    ).astype(int)


    results = {

        "ROC-AUC":
        roc_auc_score(
            y_true,
            prob
        ),


        "PR-AUC":
        average_precision_score(
            y_true,
            prob
        ),


        "Precision":
        precision_score(
            y_true,
            pred,
            zero_division=0
        ),


        "Recall":
        recall_score(
            y_true,
            pred,
            zero_division=0
        ),


        "F1":
        f1_score(
            y_true,
            pred,
            zero_division=0
        )

    }


    return results




def find_best_threshold(
    y_true,
    prob
):

    precision, recall, thresholds = precision_recall_curve(
        y_true,
        prob
    )


    f1_scores = (

        2
        *
        precision[:-1]
        *
        recall[:-1]

        /

        (
            precision[:-1]
            +
            recall[:-1]
            +
            1e-12
        )

    )


    best_index = np.argmax(
        f1_scores
    )


    best_threshold = thresholds[
        best_index
    ]


    return {

        "threshold":
        best_threshold,


        "precision":
        precision[best_index],


        "recall":
        recall[best_index],


        "F1":
        f1_scores[best_index]

    }