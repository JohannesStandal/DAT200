import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.datasets import load_iris
iris = load_iris()

# Extract numpy arrays
X = iris.data # features
y = iris.target # labels

# Convert to pandas dataframe
df = pd.DataFrame(data=X, columns=iris.feature_names)
df['label'] = pd.Series(iris.target_names[y], dtype='category')

# Make a pairplot with seaborn
palette = {'setosa': 'orange', 'versicolor': 'green', 'virginica': 'purple'}
g = sns.pairplot(df, vars = df.columns[0:4], hue="label", palette=palette)
plt.show()
