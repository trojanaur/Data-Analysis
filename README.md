# Data-Analysis
Fertilizer Data Analysis and Prediction

Overview

This code performs an exploratory data analysis on a fertilizer dataset, visualizes the distribution of fertilizer quantities, and trains a linear regression model to predict fertilizer quantities based on various features.

Dataset

The dataset used in this analysis is fertilizer_data.csv, which contains information about fertilizers, including their quantities, micronutrients, and regions.

Analysis Steps

Data Loading and Cleaning: The code loads the dataset, prints the first few rows, and summarizes the data using info() and describe() methods. It then fills missing values with the mean of each column.
Data Preprocessing: The code scales the Quantity column using StandardScaler from scikit-learn.
Data Visualization: The code creates a histogram to visualize the distribution of fertilizer quantities.
Filtering and Grouping: The code filters the data to include only fertilizers with Zinc and Sulphur micronutrients in India, groups the data by fertilizer, and calculates the sum of quantities for each fertilizer.
Correlation Analysis: The code calculates the correlation matrix for the filtered data and visualizes it using a heatmap from seaborn.
Model Training and Evaluation: The code trains a linear regression model on the filtered data, evaluates its performance using mean squared error, and performs a grid search to optimize the model's hyperparameters.
Additional Dataset

The code also loads an additional dataset case_study_data.csv, which is not used in the analysis.

Libraries Used

pandas for data manipulation and analysis
NumPy for numerical computations
Matplotlib and seaborn for data visualization
scikit-learn for machine learning tasks (preprocessing, model selection, and evaluation)
Running the Code

To run this code, simply execute the Python script in a suitable environment with the required libraries installed. Make sure to replace the fertilizer_data.csv and case_study_data.csv files with your own datasets.




