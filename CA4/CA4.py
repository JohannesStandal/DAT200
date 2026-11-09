# Data handling
import numpy as np
import pandas as pd

# Visualisation
import matplotlib.pyplot as plt

# Preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

# Pipelines and dimension reduction
from sklearn.pipeline import Pipeline

# Models
from xgboost import XGBClassifier

# Model selection
from sklearn.model_selection import train_test_split, GridSearchCV

# Evaluation
from sklearn.metrics import f1_score, accuracy_score, ConfusionMatrixDisplay, confusion_matrix

# Read data
df_train = pd.read_csv("CA4/trainCA4DAT200.csv")
df_submission = pd.read_csv("CA4/testCA4DAT200.csv")

# Dropping correlated values
df_train = df_train.drop(['DayMinutes', 'EveMinutes', 'NightMinutes', 'IntlMinutes'], axis=1)
df_submission =df_submission.drop(['DayMinutes', 'EveMinutes', 'NightMinutes', 'IntlMinutes'], axis=1)

# Split data
x_train = df_train.drop(columns=["Churn", "CustomerId"])
y_train = df_train["Churn"]

X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train, test_size=0.4, random_state=42, stratify=y_train) 

# Categorical features
categorical_features = [
    "State",
    "AreaCode",
    "InternationalPlan",
    "VoiceMailPlan"
]

# Getting the columns with numerical data for preprocessing
numerical_features = []
for col in X_train.columns:
    if col not in categorical_features:
        numerical_features.append(col)

# Numerical preprocessing with no scaling
numerical_pipeline_no_scaling = Pipeline([
    # Some models dont require scaling
    ('imputer', SimpleImputer(strategy='median')),
])

# Categorical preprocessing
categorical_pipeline = Pipeline([
    # One hot encoding because we have no ordinal features
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False))
])

# Preproccesor without scaling
preprocessor_no_scaling = ColumnTransformer([
    ('numerical', numerical_pipeline_no_scaling, numerical_features),
    ('categorical', categorical_pipeline, categorical_features)
])

# XGBoost
xgb_pipeline = Pipeline([
    ('preprocessor', preprocessor_no_scaling),
    ('classifier', XGBClassifier(random_state=42))
])

xgb_param_grid = {
    'classifier__n_estimators': [20, 50, 100, 200],
    'classifier__learning_rate': [0.001, 0.01, 0.1],
    'classifier__max_depth': [3, 5, 7],
    'classifier__subsample': [0.7],
    'classifier__gamma': [0, 3, 5, 10],
    'classifier__reg_lambda': [0, 3, 5, 10],
}

xgb_grid = GridSearchCV(
    xgb_pipeline,
    param_grid=xgb_param_grid,
    cv=5,
    scoring='f1_macro'
)

# Fit the grid search to the training data
xgb_grid.fit(X_train, Y_train)
print(xgb_grid.best_score_)
xgb_best_model = xgb_grid.best_estimator_


# Evaluate the model on the test set and print the accuracy, F1 score, and plot confusion matrix
print(f"Evaluating XGBoost") 
y_pred = xgb_best_model.predict(X_test)

# Print scores
print(f"accuracy: {accuracy_score(Y_test, y_pred)}")
print(f"F1 score: {f1_score(Y_test, y_pred, average='macro')}")

# Confusion matrix
cm = confusion_matrix(Y_test, y_pred)

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=['No Churn', 'Churn']
).plot()

plt.title("XGBoost")
plt.show()

# Create submission file
customerIds = df_submission['CustomerId']
x_input = df_submission.drop(['CustomerId'], axis=1)

# Predict targets
y_pred = xgb_best_model.predict(x_input)

# create submission dataframe and save as CSV file
submission = pd.DataFrame({
    "CustomerId": customerIds,
    "Churn": y_pred
})
submission.to_csv("submissionCA4fromPythonScript.csv", index=False)
