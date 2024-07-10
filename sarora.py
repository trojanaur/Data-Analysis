import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import seaborn as sns

data = pd.read_csv('fertilizer_data.csv')
print(data.head())
print(data.info())
print(data.describe())

data.fillna(data.mean(), inplace=True)
scaler = StandardScaler()
data[['Quantity']] = scaler.fit_transform(data[['Quantity']])

plt.hist(data['Quantity'], bins=50)
plt.xlabel('Quantity')
plt.ylabel('Frequency')
plt.title('Distribution of Quantity')
plt.show()

india_data = data[(data['Micronutrient'].isin(['Zinc', 'Sulphur'])) & (data['Region'] == 'India')]
fertilizer_quantities = india_data.groupby('Fertilizer')['Quantity'].sum().reset_index()
fertilizer_quantities = fertilizer_quantities.sort_values('Quantity', ascending=False)
print(fertilizer_quantities.head())

plt.bar(fertilizer_quantities['Fertilizer'], fertilizer_quantities['Quantity'])
plt.xlabel('Fertilizer')
plt.ylabel('Quantity')
plt.title('Top Fertilizers with Zinc and Sulphur Micronutrients in India')
plt.show()

correlation_matrix = india_data.corr()
print(correlation_matrix)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()

X = india_data.drop('Quantity', axis=1)
y = india_data['Quantity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse:.2f}')

param_grid = {'n_jobs': [-1], 'normalize': [True, False]}
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
print(f'Best Parameters: {grid_search.best_params_}')
print(f'Best Score: {grid_search.best_score_:.2f}')

case_study_data = pd.read_csv('case_study_data.csv')