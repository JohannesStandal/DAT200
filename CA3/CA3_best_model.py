import sklearn as sk
import pandas as pd

# Reading the data
df_train = pd.read_csv("CA3/train.csv")
df_kaggle = pd.read_csv("CA3/test.csv")

# Feature selection
x_train = df_train.drop(['quality', 'Id', 'chlorides', 'residual sugar'], axis=1)
y_train = df_train['quality']

x_kaggle = df_kaggle.drop(['Id', 'chlorides', 'residual sugar'], axis=1)

# Convert dataframe to np arrays
X_train = x_train.to_numpy()
Y_train = y_train.to_numpy()

X_kaggle = x_kaggle.to_numpy()

# Scaling training data 
# The code is here so that you can replace a model with very little effort
# Our program declared randomforest as best model, wich does not use the scaled data
scaler = sk.preprocessing.StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_kaggle_scaled = scaler.transform(X_kaggle)

# Setting up the model
"""
Output from notebook code:
-----------------------------------------------------------------
Best model was Forest with an mean accuraccy of 76.94%
Here are the parameters used:
{'criterion': 'entropy', 'max_depth': None, 'n_estimators': 300}
"""

model = sk.ensemble.RandomForestClassifier(
    criterion = "entropy",
    max_depth = None,
    n_estimators = 300, 
    random_state=42
)

# Training the model
model.fit(X_train, Y_train)

# Predict targets and create a dataframe and CSV file
y_pred = model.predict(X_kaggle)
y_pred = pd.DataFrame(y_pred, columns=["quality"])
y_pred.index.name = "Id"
y_pred[['quality']].to_csv("submissionFromScript.csv")