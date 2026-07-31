# ============================================================
# EARLY STUDENT DROPOUT AND ACADEMIC SUCCESS PREDICTION
# ============================================================

from ucimlrepo import fetch_ucirepo

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    StratifiedKFold
)

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    LabelEncoder
)

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from sklearn.utils.class_weight import compute_sample_weight

from xgboost import XGBClassifier


# ============================================================
#  LOAD DATASET
# ============================================================

predict_students_dropout_and_academic_success = fetch_ucirepo(
    id=697
)


# ============================================================
# COLUMNS TO REMOVE
# ============================================================

semester_columns = [

    'Curricular units 1st sem (credited)',

    'Curricular units 1st sem (enrolled)',

    'Curricular units 1st sem (evaluations)',

    'Curricular units 1st sem (approved)',

    'Curricular units 1st sem (grade)',

    'Curricular units 1st sem (without evaluations)',

    'Curricular units 2nd sem (credited)',

    'Curricular units 2nd sem (enrolled)',

    'Curricular units 2nd sem (evaluations)',

    'Curricular units 2nd sem (approved)',

    'Curricular units 2nd sem (grade)',

    'Curricular units 2nd sem (without evaluations)',

    'Tuition fees up to date'

]


# ============================================================
#CREATE FEATURES AND TARGET
# ============================================================

X = predict_students_dropout_and_academic_success.data.features

Y = predict_students_dropout_and_academic_success.data.targets


# Remove selected columns
X = X.drop(
    columns=semester_columns
)



Y = Y.squeeze()


print("\n====================================")
print("DATASET INFORMATION")
print("====================================")

print(
    "Features:",
    X.shape
)

print(
    "Target:",
    Y.shape
)


# ============================================================
#  CHECK TARGET CLASSES
# ============================================================

print("\n====================================")
print("TARGET CLASSES")
print("====================================")

print(
    Y.value_counts()
)


# ============================================================
#LABEL ENCODING
# ============================================================



label_encoder = LabelEncoder()

label_encoder.fit(Y)


print("\n====================================")
print("LABEL ENCODING")
print("====================================")

for number, label in enumerate(
    label_encoder.classes_
):

    print(
        label,
        "->",
        number
    )


# ============================================================
#  SPLIT DATA
# ============================================================



X_train, X_temp, Y_train, Y_temp = train_test_split(

    X,

    Y,

    test_size=0.30,

    random_state=42,

    stratify=Y

)


X_val, X_test, Y_val, Y_test = train_test_split(

    X_temp,

    Y_temp,

    test_size=0.50,

    random_state=42,

    stratify=Y_temp

)


print("\n====================================")
print("DATA SPLIT")
print("====================================")

print(
    "Training:",
    X_train.shape
)

print(
    "Validation:",
    X_val.shape
)

print(
    "Testing:",
    X_test.shape
)


# ============================================================
# DEFINE CATEGORICAL FEATURES
# ============================================================

categorical_features = [

    'Marital Status',

    'Application mode',

    'Application order',

    'Course',

    'Daytime/evening attendance',

    'Previous qualification',

    'Nacionality',

    "Mother's qualification",

    "Father's qualification",

    "Mother's occupation",

    "Father's occupation",

    'Displaced',

    'Educational special needs',

    'Debtor',

    'Gender',

    'Scholarship holder',

    'International'

]


# ============================================================
# DEFINE NUMERICAL FEATURES
# ============================================================

numerical_features = [

    col

    for col in X.columns

    if col not in categorical_features

]


print("\n====================================")
print("FEATURE INFORMATION")
print("====================================")

print(
    "Number of categorical features:",
    len(categorical_features)
)

print(
    "Number of numerical features:",
    len(numerical_features)
)


# ============================================================
# PREPROCESSING PIPELINE
# ============================================================

preprocessor = ColumnTransformer(

    transformers=[

        (

            'num',

            StandardScaler(),

            numerical_features

        ),

        (

            'cat',

            OneHotEncoder(
                handle_unknown='ignore'
            ),

            categorical_features

        )

    ]

)


# ============================================================
#  LOGISTIC REGRESSION
# ============================================================

logistic_pipeline = Pipeline(

    steps=[

        (

            'preprocessor',

            preprocessor

        ),

        (

            'classifier',

            LogisticRegression(

                max_iter=1000

            )

        )

    ]

)


logistic_pipeline.fit(

    X_train,

    Y_train

)


Y_val_pred_logistic = (

    logistic_pipeline.predict(

        X_val

    )

)


print("\n====================================")
print("# LOGISTIC REGRESSION #")
print("====================================")

print(

    classification_report(

        Y_val,

        Y_val_pred_logistic

    )

)


# ============================================================
# BALANCED LOGISTIC REGRESSION
# ============================================================

logistic_balanced_pipeline = Pipeline(

    steps=[

        (

            'preprocessor',

            preprocessor

        ),

        (

            'classifier',

            LogisticRegression(

                max_iter=1000,

                class_weight='balanced'

            )

        )

    ]

)


logistic_balanced_pipeline.fit(

    X_train,

    Y_train

)


Y_val_pred_logistic_balanced = (

    logistic_balanced_pipeline.predict(

        X_val

    )

)


print("\n====================================")
print("# LOGISTIC REGRESSION BALANCED #")
print("====================================")

print(

    classification_report(

        Y_val,

        Y_val_pred_logistic_balanced

    )

)


# ============================================================
#DECISION TREE - BALANCED
# ============================================================

decision_tree_pipeline = Pipeline(

    steps=[

        (

            'preprocessor',

            preprocessor

        ),

        (

            'classifier',

            DecisionTreeClassifier(

                class_weight='balanced',

                random_state=42

            )

        )

    ]

)


decision_tree_pipeline.fit(

    X_train,

    Y_train

)


Y_val_pred_tree = (

    decision_tree_pipeline.predict(

        X_val

    )

)


print("\n====================================")
print("# DECISION TREE - BALANCED #")
print("====================================")

print(

    classification_report(

        Y_val,

        Y_val_pred_tree

    )

)


# ============================================================
#  RANDOM FOREST - BALANCED
# ============================================================

random_forest_pipeline = Pipeline(

    steps=[

        (

            'preprocessor',

            preprocessor

        ),

        (

            'classifier',

            RandomForestClassifier(

                n_estimators=100,

                class_weight='balanced',

                random_state=42,

                n_jobs=-1

            )

        )

    ]

)


random_forest_pipeline.fit(

    X_train,

    Y_train

)


Y_val_pred_rf = (

    random_forest_pipeline.predict(

        X_val

    )

)


print("\n====================================")
print("# RANDOM FOREST - BALANCED #")
print("====================================")

print(

    classification_report(

        Y_val,

        Y_val_pred_rf

    )

)


# ============================================================
#  ORIGINAL XGBOOST
# ============================================================



Y_train_encoded = label_encoder.transform(

    Y_train

)


Y_val_encoded = label_encoder.transform(

    Y_val

)


Y_test_encoded = label_encoder.transform(

    Y_test

)


xgboost_pipeline = Pipeline(

    steps=[

        (

            'preprocessor',

            preprocessor

        ),

        (

            'classifier',

            XGBClassifier(

                n_estimators=100,

                max_depth=6,

                learning_rate=0.1,

                random_state=42,

                n_jobs=-1,

                eval_metric='mlogloss'

            )

        )

    ]

)


xgboost_pipeline.fit(

    X_train,

    Y_train_encoded

)


Y_val_pred_xgb_encoded = (

    xgboost_pipeline.predict(

        X_val

    )

)


Y_val_pred_xgb = (

    label_encoder.inverse_transform(

        Y_val_pred_xgb_encoded.astype(int)

    )

)


print("\n====================================")
print("# XGBOOST #")
print("====================================")

print(

    classification_report(

        Y_val,

        Y_val_pred_xgb

    )

)


# ============================================================
# BALANCED XGBOOST
# ============================================================

xgb_sample_weights = (

    compute_sample_weight(

        class_weight='balanced',

        y=Y_train_encoded

    )

)


xgboost_balanced_pipeline = Pipeline(

    steps=[

        (

            'preprocessor',

            preprocessor

        ),

        (

            'classifier',

            XGBClassifier(

                n_estimators=100,

                max_depth=6,

                learning_rate=0.1,

                random_state=42,

                n_jobs=-1,

                eval_metric='mlogloss'

            )

        )

    ]

)


xgboost_balanced_pipeline.fit(

    X_train,

    Y_train_encoded,

    classifier__sample_weight=xgb_sample_weights

)


Y_val_pred_xgb_balanced_encoded = (

    xgboost_balanced_pipeline.predict(

        X_val

    )

)


Y_val_pred_xgb_balanced = (

    label_encoder.inverse_transform(

        Y_val_pred_xgb_balanced_encoded.astype(int)

    )

)


print("\n====================================")
print("# XGBOOST - BALANCED #")
print("====================================")

print(

    classification_report(

        Y_val,

        Y_val_pred_xgb_balanced

    )

)


# ============================================================
# HYPERPARAMETER TUNING
# ============================================================

param_distributions = {

    'classifier__n_estimators': [

        100,

        200,

        300,

        500

    ],

    'classifier__max_depth': [

        3,

        4,

        5,

        6,

        8,

        10

    ],

    'classifier__learning_rate': [

        0.01,

        0.05,

        0.1,

        0.2

    ],

    'classifier__subsample': [

        0.7,

        0.8,

        0.9,

        1.0

    ],

    'classifier__colsample_bytree': [

        0.7,

        0.8,

        0.9,

        1.0

    ]

}


cv_strategy = StratifiedKFold(

    n_splits=5,

    shuffle=True,

    random_state=42

)


tuning_pipeline = Pipeline(

    steps=[

        (

            'preprocessor',

            preprocessor

        ),

        (

            'classifier',

            XGBClassifier(

                random_state=42,

                n_jobs=-1,

                eval_metric='mlogloss'

            )

        )

    ]

)


random_search = RandomizedSearchCV(

    estimator=tuning_pipeline,

    param_distributions=param_distributions,

    n_iter=30,

    scoring='f1_macro',

    cv=cv_strategy,

    verbose=1,

    random_state=42,

    n_jobs=-1

)


random_search.fit(

    X_train,

    Y_train_encoded

)


print("\n====================================")
print("# BEST XGBOOST PARAMETERS #")
print("====================================")

print(

    random_search.best_params_

)


print("\nBest CV Macro F1:")

print(

    random_search.best_score_

)


# ============================================================
#TUNED XGBOOST VALIDATION RESULTS
# ============================================================

tuned_xgboost = (

    random_search.best_estimator_

)


Y_val_pred_tuned_xgb_encoded = (

    tuned_xgboost.predict(

        X_val

    )

)


Y_val_pred_tuned_xgb = (

    label_encoder.inverse_transform(

        Y_val_pred_tuned_xgb_encoded.astype(int)

    )

)


print("\n====================================")
print("# TUNED XGBOOST VALIDATION RESULTS #")
print("====================================")

print(

    classification_report(

        Y_val,

        Y_val_pred_tuned_xgb

    )

)


# ============================================================
#  FINAL MODEL SELECTION
# ============================================================

print("\n====================================")
print("# FINAL MODEL SELECTION #")
print("====================================")

print(

    "The original XGBoost model is selected "

    "as the final model based on the validation results."

)


# ============================================================
# COMBINE TRAINING + VALIDATION DATA
# ============================================================


X_train_final = pd.concat(

    [

        X_train,

        X_val

    ]

)


Y_train_final = pd.concat(

    [

        Y_train,

        Y_val

    ]

)


print("\n====================================")
print("# FINAL MODEL TRAINING DATA #")
print("====================================")

print(

    "Final training data:",

    X_train_final.shape

)


# ============================================================
#ENCODE FINAL TRAINING TARGET
# ============================================================

Y_train_final_encoded = (

    label_encoder.transform(

        Y_train_final

    )

)


print(

    "Final target classes:",

    np.unique(

        Y_train_final_encoded

    )

)


# ============================================================
# CREATE FINAL XGBOOST MODEL
# ============================================================

final_xgboost_pipeline = Pipeline(

    steps=[

        (

            'preprocessor',

            preprocessor

        ),

        (

            'classifier',

            XGBClassifier(

                n_estimators=100,

                max_depth=6,

                learning_rate=0.1,

                random_state=42,

                n_jobs=-1,

                eval_metric='mlogloss'

            )

        )

    ]

)


# ============================================================
#  TRAIN FINAL XGBOOST MODEL
# ============================================================

final_xgboost_pipeline.fit(

    X_train_final,

    Y_train_final_encoded

)


print("\n====================================")
print("# FINAL XGBOOST MODEL TRAINED #")
print("====================================")


# ============================================================
#  PREDICT ON UNSEEN TEST SET
# ============================================================

Y_test_pred_encoded = (

    final_xgboost_pipeline.predict(

        X_test

    )

)


# ============================================================
#  CONVERT PREDICTIONS BACK TO LABELS
# ============================================================

Y_test_pred = (

    label_encoder.inverse_transform(

        Y_test_pred_encoded.astype(int)

    )

)


# ============================================================
#  ACTUAL TEST LABELS
# ============================================================



Y_test_actual = Y_test


# ============================================================
#  FINAL MODEL TEST RESULTS
# ============================================================

print("\n====================================")
print("# FINAL MODEL TEST RESULTS #")
print("====================================")


print(

    classification_report(

        Y_test_actual,

        Y_test_pred,

        labels=label_encoder.classes_,

        target_names=label_encoder.classes_

    )

)


# ============================================================
# FINAL TEST ACCURACY
# ============================================================

final_accuracy = accuracy_score(

    Y_test_actual,

    Y_test_pred

)


print(

    "Final Test Accuracy:",

    round(

        final_accuracy,

        4

    )

)


# ============================================================
#  FINAL CONFUSION MATRIX
# ============================================================

print("\n====================================")
print("# FINAL CONFUSION MATRIX #")
print("====================================")


ConfusionMatrixDisplay.from_predictions(

    Y_test_actual,

    Y_test_pred,

    labels=label_encoder.classes_

)


plt.title(

    "Final XGBoost Model - Unseen Test Set"

)


plt.xlabel(

    "Predicted Label"

)


plt.ylabel(

    "Actual Label"

)


plt.tight_layout()


plt.show()