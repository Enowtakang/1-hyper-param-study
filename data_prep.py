import pandas as pd


# Load dataset
data = pd.read_csv('ydata.csv')
print(data.shape)

# Drop the YEAR column
data = data.drop('YEAR', axis=1)
print(data.shape)

# Remove all rows containing zeros
data = data.loc[(data != 0).all(axis=1)]
print(data.shape)

# Save prepared data to a separate CSV file
ydata_processed = data.to_csv(
    'data.csv',
    index=None)
