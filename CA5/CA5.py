# Data handling
import numpy as np
import pandas as pd

# Preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

# Pipelines and dimension reduction
from sklearn.pipeline import Pipeline

# Model
from sklearn.ensemble import RandomForestClassifier

def cleaning(df):
    df = df.copy()
    original_size = len(df)

    # Features that can be clipped between 1-10
    clipped_features  = [
        'sleep_quality_score', 
        'social_connection_score', 
        'chronic_stress_score'
    ]

    # Clip salvageable features
    for col in clipped_features:
        df[col] = df[col].clip(1, 10)


    # Valid range for features 
    feature_ranges = {
        'age': (18, 90), # Specified in data description 
        'sleep_duration_hrs': (0, 24), # Good luck sleeping more than 24 hours a day
        'cardio_mins_wk': (0, 1100), # assume less than 2,6 hours a day 
        'water_intake_liters': (0, 6),
        'screen_time_recreational_hrs': (0, 20), # you need at least 4 hours sleep
        'social_media_hrs': (0, 20), 
        'blood_pressure_sys': (70, 200), 
        'diet_ultra_processed_pct': (0, 100), # percent 0-100
        'diet_omega3_freq': (0, 7),  #  Days per week of consuming brain-healthy foods
        'alcohol_drinks_wk': (0, 60),
        'novel_learning_hrs_wk': (0, 20), # you need at least 4 hours sleep
    }

    # Remove observations outside valid ranges
    for col, (lower, upper) in feature_ranges.items():
        l2 = len(df)

        # Keep NAN values and values inside the value range
        df = df[ df[col].isna() | df[col].between(lower, upper)]
        l1 = len(df)

        print(f"Removed {l2-l1} ({(l2-l1) / original_size:.2%}) rows out of range ({lower}, {upper}) from {col}. ")

    removed = original_size - len(df)

    print(f"Removed {removed} rows ({removed / original_size:.2%})")

    return df

# Read CSV files 
df_train = pd.read_csv('CA5/train.csv')
df_submission = pd.read_csv('CA5/test.csv')

df_train = cleaning(df_train)

# Data split
x_train = df_train.drop(columns=["target", "id"])
y_train = df_train["target"]

# Naming the columns with categorical data for preprocessing 
categorical_features = [
    "smoking_status"
]

# Getting the columns with numerical data for preprocessing
numerical_features = []
for col in x_train.columns:
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

# Random forest classifier pipeline
RF_classifier_pipeline = Pipeline([
    ('preprocessor', preprocessor_no_scaling),
    ('classifier', RandomForestClassifier(
        random_state=42, 
        max_depth = None, 
        min_samples_leaf = 3, 
        n_estimators = 200
    ))
])

# Create bins
n_bins = 25

# Recreate target bins using all available training data
y_full_binned = pd.qcut(
    y_train,
    q=n_bins,
    labels=False,
    duplicates='drop'
)

# Calculate new bin medians
final_bin_centers = y_train.groupby(y_full_binned).median().to_dict()
RF_model = RF_classifier_pipeline

# Train on all available data
RF_model.fit(x_train, y_full_binned)

# Get submission data and participant ID
participantIds = df_submission['id']
x_input = df_submission.drop(['id'], axis=1)

# Map predicted bins to the bin median
y_pred = RF_model.predict(x_input)
y_pred = np.array([final_bin_centers[bin] for bin in y_pred]) 

# Clip predictions
y_pred = np.clip(y_pred, 0, 30)

# create submission dataframe and save as CSV file
submission = pd.DataFrame({
    "id": participantIds,
    "target": y_pred
})
submission.to_csv("submissionCA5_fromPythonScript.csv", index=False)
