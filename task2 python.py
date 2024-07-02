

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file
file_path = 'tested.csv'
df = pd.read_csv(file_path)

# Data Cleaning
# Handle missing values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)
df.drop(columns=['Cabin'], inplace=True)  # Dropping 'Cabin' due to high number of missing values

# Remove duplicates
df.drop_duplicates(inplace=True)

# Correct data types
df['Survived'] = df['Survived'].astype('category')
df['Pclass'] = df['Pclass'].astype('category')
df['Sex'] = df['Sex'].astype('category')
df['Embarked'] = df['Embarked'].astype('category')

# Exploratory Data Analysis (EDA)
# Summary statistics
summary_stats = df.describe(include='all')
print(summary_stats)

# Visualizations
plt.figure(figsize=(12, 6))

# Distribution of Age
plt.subplot(2, 2, 1)
sns.histplot(df['Age'], kde=True, bins=30)
plt.title('Age Distribution')

# Distribution of Fare
plt.subplot(2, 2, 2)
sns.histplot(df['Fare'], kde=True, bins=30)
plt.title('Fare Distribution')

# Survival rate by Sex
plt.subplot(2, 2, 3)
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title('Survival Rate by Sex')

# Survival rate by Pclass
plt.subplot(2, 2, 4)
sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title('Survival Rate by Pclass')

plt.tight_layout()
plt.show()

# Correlation heatmap
# Exclude non-numeric columns
numeric_df = df.select_dtypes(include=['float64', 'int64'])
plt.figure(figsize=(10, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

