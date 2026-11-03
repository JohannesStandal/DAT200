# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 20:42:47 2019

@author: olto
"""


# =============================================================================
# Import modules
# =============================================================================

from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# =============================================================================
# Load data
# =============================================================================

# Load the California housing dataset from scikit learn

data = datasets.fetch_california_housing()


# =============================================================================
# Inspect and visualise raw data
# =============================================================================

# Put data into pandas dataframe to access plotting functions
data_df = pd.DataFrame(data['data'])
data_df.columns = data['feature_names']


# Plot histograms
data_df.hist(figsize=(7, 7))
plt.show()

# Get descriptive statistics
descr_stats = data_df.describe()
print(descr_stats)


# Make box plots to visualise distribution
data_df.plot(kind='box', 
             subplots=True, 
             layout=(2, 8), 
             sharex=False, 
             sharey=False,
             figsize=(10, 10))            
plt.show()



# Box plot - all features in one plot
plt.figure(0,figsize=(10, 10))
data_df.boxplot()
plt.show()


# =============================================================================
# Violin plot
# =============================================================================

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 6))

# Draw violin plot
sns.violinplot(data=data_df, palette="Set3", bw=.2, cut=1, linewidth=1)

#ax.set(ylim=(-.7, 1.05))
sns.despine(left=True, bottom=True)
plt.show()



# =============================================================================
# Scale data
# =============================================================================
X = data['data']
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0, ddof=1)


X_sc = (X - np.mean(X, axis=0)) / np.std(X, axis=0, ddof=1)
X_sc_mean = np.mean(X_sc, axis=0)
X_sc_std = np.std(X_sc, axis=0, ddof=1)


# Put data into pandas dataframe to access plotting functions
data_df_sc = pd.DataFrame(X_sc)
data_df_sc.columns = data['feature_names']


# # Plot histograms for scaled features
# data_df_sc.hist()
# plt.show()


# Box plot - all scaled features in one plot
plt.figure(1, figsize=(10, 10))
data_df_sc.boxplot()
plt.show()


# Set up the matplotlib figure for violin plot
f, ax = plt.subplots(figsize=(15, 8))

# Draw violin plot
sns.violinplot(data=data_df_sc, palette="Set3", bw=.2, cut=1, linewidth=1)

#ax.set(ylim=(-.7, 1.05))
sns.despine(left=True, bottom=True)
plt.show()



# =============================================================================
# Splitting the data into train and test sets
# =============================================================================
X_train = data.data[:350][:]
X_test = data.data[350:][:]


# Check descriptive statistics of X_train
X_train_df = pd.DataFrame(X_train)
X_train_df.columns = data['feature_names']
X_train_descr_stats = X_train_df.describe()


# Check descriptive statistics of X_test
X_test_df = pd.DataFrame(X_test)
X_test_df.columns = data['feature_names']
X_test_descr_stats = X_test_df.describe()


# =============================================================================
# Scale the data, both training data and test data
# =============================================================================

# Scale X_train and then scale X_test with parameters (mean & std) of X_train
X_train_sc = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0, ddof=1)
X_test_sc = (X_test - np.mean(X_train, axis=0)) / np.std(X_train, axis=0, ddof=1)    


# Double-check mean and std of the scaled X_train
X_train_sc_mean = np.mean(X_train_sc, axis=0)
X_train_sc_std = np.std(X_train_sc, axis=0, ddof=1)


# Double-check mean and std of the scaled X_test
X_test_sc_mean = np.mean(X_test_sc, axis=0)
X_test_sc_std = np.std(X_test_sc, axis=0, ddof=1)





