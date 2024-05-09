import pandas as pd
from numpy import pi

data = pd.read_csv("RELU_MZI_training_data.csv", header=None)
print(data.describe())
# Generate a correlation matrix.
print(data.corr())