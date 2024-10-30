# Car_price_Prediction_With_ML-
Let's break down the code and explain the workflow step by step:

Import Libraries:

Import the necessary Python libraries, including Pandas, NumPy, Matplotlib, scikit-learn modules for data manipulation, visualization, and machine learning.
Load the Dataset:

Use pd.read_csv to load the car price dataset from the specified file path into a Pandas DataFrame.
Data Preprocessing:

Use one-hot encoding to convert categorical variables (Fuel_Type, Selling_type, Transmission) into binary format (0 or 1) so that they can be used in the machine learning model. The drop_first=True argument avoids multicollinearity.
Data Splitting:

Split the data into features (X) and the target variable (y). X contains all columns except "Car_Name" and "Selling_Price," and y contains the "Selling_Price" column, which is what we want to predict.
Train-Test Split:

Split the data into a training set (80%) and a testing set (20%) using train_test_split. This allows us to evaluate the model's performance on unseen data.
Model Selection:

Create a Random Forest Regressor model. This is a machine learning model for regression tasks. The n_estimators parameter specifies the number of decision trees in the ensemble, and random_state ensures reproducibility.
Model Training:

Fit the Random Forest Regressor model to the training data using model.fit(X_train, y_train).
Model Prediction:

Use the trained model to make predictions on the test data. The predicted values are stored in y_pred.
Model Evaluation:

Calculate various regression metrics to assess the model's performance on the test data, including Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and the R-squared (R2) score. These metrics provide information about how well the model's predictions match the actual car prices.

Visualization
