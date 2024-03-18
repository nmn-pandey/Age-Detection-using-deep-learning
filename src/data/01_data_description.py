import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Setting global variable for the ipynb file
data_dir = '../data'

# Loading the training and testing dataset
train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))

# Check data quality
print(train_df.info())
print(test_df.info())

# Describe the shape, head, and summary of the training dataset
print(train_df.shape)
print(train_df.head())
print(train_df.describe())

# Describe the shape, head, and summary of the testing dataset
print(test_df.shape)
print(test_df.head())
print(test_df.describe())

# Check for missing values
print("\nData quality check:")
print(train_df.isnull().sum())
print(test_df.isnull().sum())

# Data distribution visualization
plt.figure(figsize=(10, 6))
sns.histplot(train_df['age'], bins=20, kde=True)
plt.title('Distribution of Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Box plot to visualize outliers
plt.figure(figsize=(10, 6))
sns.boxplot(x=train_df['age'])
plt.title('Boxplot of Age')
plt.xlabel('Age')
plt.show()

