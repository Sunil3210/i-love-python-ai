import pandas as pd

# Important working with data-frames
# 1) Import a csv file
df = pd.read_csv("D:/PythonProjects/DataAnalytics/DataAnalytics/files/data.csv") 
d1 = pd.read_csv("D:/PythonProjects/DataAnalytics/DataAnalytics/files/data.csv",usecols=['carat', 'cut'])   # Import sepcific columns


# 2) Exploring data-set
df.head()       # first five rows
df.tail()       # last five rows
df.sample(5)    # random sample of rows
df.shape        # number of rows/columns in a tuple
df.describe()   # calculates measures of central tendency
df.info()       # memory footprint and datatypes

# 3) Add a new column to a dataset
df['price_rupee'] = df.price * 60
df['price_rupee']
# 4) Filtering data-frames with conditional logic
filtered_data = df[df.cut == 'Premium']

# 5) Sorting a dataframew
df.sort_values('price', axis=0, ascending=False)

# 6) Apply a function to every row in a column
def price_euro(price):
    cp = price * 80
    return cp

df['price_euro'] = df.price.apply(price_euro)

# 7) Add a new column with conditional logic
import numpy as np
df['costly'] = np.where(df['price']>=300.00, True, False)  # Use numpy
df['costly'] = np.where(df['price']>=300.00, 'Costly', 'Common')  # Use numpy

# 8) Converting a dataframe to numpy array
df.values
#df.as_matrix

# 9) Combine data-frames with concatenation
pd.concat([df, filtered_data], axis=0) # concat columns vertically
pd.concat([df, filtered_data], axis=1) # concat columns horizontally:


# 10) Combine dataframes based on an index key
merged_df = filtered_data.merge(df, how='inner', on='price')

# 11) Finding NaNs in a dataframe
df.isnull().sum()       # list the NaN count of each column
df.isnull().sum().sum() # Total number of NaN present


# 12) Filling NaNs or Missing Data
df.price = df.price.fillna('1234')

df = df.dropna(axis=0)   # drop any row missing data across the entire dataframe



# 13) Extracting Features by Grouping Columns
# group diamonds by cut , then calculate the mean for price in each group.
df.groupby('cut')['price'].apply(lambda x: np.mean(x))
# see the count of cut
df.groupby('cut')['price'].apply(lambda x: x.count())
