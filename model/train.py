# Import libraries
import pandas as pd
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import pickle
import gzip#
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error

def explore_df(df):
    """ This function reads a CSV file, displays basic information about 
    the df, and returns descriptive statistics of the data.

    Parameters:
        file_path (str): Path to the CSV file.

    Returns:
        df (DataFrame): The DataFrame read from the CSV file.
    """
    # Read the CSV file into a DataFrame
    #df = pd.read_csv(file_path)

    # Display the first lines of the df
    #print(df.head().T)
    #print()

    # Display the number of proporties and features
    #num_properties, num_features = df.shape
    #print(f"There are {num_properties} proporties and {num_features} features.\n")

    # Display the features:
    #print(f"The features are: {', '.join([str(feature)for feature in df.columns])}.\n")
     
    # Display the descriptive statistics
    #print(df.describe(include="all").T)
    #print()

    # return the df
    return df

# Specify the file path
df = pd.read_csv(r"data\cleaned_properties.csv")
# Call the function to explore the data
explore_df(df)


def filter_houses(df):
    """Filter out the DataFrame for values where the property_type is 'HOUSE'
    and the subproperty_type is not 'APARTMENT_BLOCK'.

    Args:
        df (DataFrame): Input DataFrame. 

    Returns:
        Dataframe: Filtered DataFrame containing only HOUSE properties.
    """    
    # Filter out the DataFrame for values APARTMENT and APARTMENT_BLOCK
    df_house = df[(df["property_type"] == "HOUSE") & (df['subproperty_type'] != 'APARTMENT_BLOCK')]
    # df_house.info()

    return df_house

# Call the filter_houses function and pass your DataFrame as an argument
df_house = filter_houses(df)


def prepare_data(df_house):
    """Prepare the data for machine learning by splitting it into features (X) and target variable (y),
    and then splitting it into training and test sets.

    Parameters:
        df_house (DataFrame): DataFrame containing the subset of houses data.

    Returns:
        X_train (DataFrame): Features for training.
        X_test (DataFrame): Features for testing.
        y_train (Series): Target variable for training.
        y_test (Series): Target variable for testing.
    """    
    # Name X and y
    X = df_house.drop(columns=['price', 'property_type', 'subproperty_type', 'region', 'locality', 'construction_year', 'nbr_frontages', 'fl_floodzone', 'cadastral_income'])
    y = df_house['price']

    # Split the data into training and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
    
    # X_train.info()
    # Display the features:
    #print(f"The features of df_house are:\n {', '.join([str(feature)for feature in X_train.columns])}.\n")

    return X_train, X_test, y_train, y_test

# Call prepare_data and pass df_house as an argument:
X_train, X_test, y_train, y_test = prepare_data(df_house)


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd

def preprocess_data(X_train):
    """Preprocesses training data including imputation, encoding, and scaling.

    Args:
        X_train (pandas DataFrame): Input training DataFrame.

    Returns:
        pandas DataFrame: Preprocessed training DataFrame.
    """    
    # Separate numerical and categorical columns
    numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X_train.select_dtypes(include=['object']).columns

    # Define preprocessing steps for numerical and categorical data
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    # Fit and transform the preprocessing steps on training data
    X_train_processed = preprocessor.fit_transform(X_train)

    # Convert the processed data into DataFrames
    X_train_processed = pd.DataFrame(X_train_processed, columns=numeric_cols.tolist() +
                                     preprocessor.named_transformers_['cat']
                                     .named_steps['onehot'].get_feature_names_out(categorical_cols).tolist())
    
    return X_train_processed, preprocessor

# Preprocess training data and obtain the preprocessor object
X_train_processed, preprocessor = preprocess_data(X_train)
# Save the preprocessor to a file
with gzip.open(r"model\preprocessor.pkl", 'wb') as f:
    pickle.dump(preprocessor, f)


def preprocess_data_for_test(X_test, preprocessor):
    """Preprocesses test data including imputation, encoding, and scaling.

    Parameters:
        X_test (pandas DataFrame): Input test DataFrame.
        preprocessor (sklearn ColumnTransformer): Fitted preprocessor used to transform the test data.

    Returns:
        pandas DataFrame: Preprocessed test DataFrame.
    """
    # Transform the test data using the fitted preprocessor
    X_test_processed = preprocessor.transform(X_test)

    # Get the column names for the transformed data
    numeric_cols = X_test.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X_test.select_dtypes(include=['object']).columns

    transformed_columns = numeric_cols.tolist() + \
                          preprocessor.named_transformers_['cat'].named_steps['onehot'] \
                          .get_feature_names_out(categorical_cols).tolist()

    # Convert the processed data into a DataFrame
    X_test_processed = pd.DataFrame(X_test_processed, columns=transformed_columns)

    return X_test_processed


# Preprocess test data using the preprocessor object
X_test_processed = preprocess_data_for_test(X_test, preprocessor)
# Save the preprocess_data_for_test to a file
# with gzip.open('preprocessing.pkl', 'wb') as f:
    # pickle.dump(preprocess_data_for_test, f)


def train_model(X_train_processed, y_train, n_estimators=100, random_state=42):
    """Initialize and train a Random Forest Regressor model.

    Parameters:
        X_train_processed (DataFrame): Preprocessed features for training.
        y_train (Series): Target variable for training.
        n_estimators (int, optional): Number of trees in the forest. Defaults to 10.
        random_state (int, optional): Seed used by the random number generator. Defaults to 42.

    Returns:
        RandomForestRegressor: Trained Random Forest Regressor model.
    """    
    # Initialize the RFR model
    regressor = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    # Train the model using the processed X_train and y_train
    regressor.fit(X_train_processed, y_train)

    return regressor

# Call train_model
regressor = train_model(X_train_processed, y_train)

# Save the model to a file
with gzip.open(r"model\random_forest_regressor.pkl", 'wb') as f:
    pickle.dump(regressor, f)

def evaluate_model(regressor, X_train_processed, y_train, X_test_processed, y_test):
    """Predicts target variable, evaluates the regression model, and 
    prints evaluation metrics.

    Parameters:
        regressor (RandomForestRegressor): Trained RandomForestRegressor model.
        X_train_processed (DataFrame): Preprocessed features for training.
        y_train (Series): Target variable for training.
        X_test_processed (DataFrame): Preprocessed features for testing.
        y_test (Series): Target variable for testing.

    Returns:
        Dataframe: DataFrame containing actual and predicted values for the test set.
    """    
    # Predictions
    y_train_pred = regressor.predict(X_train_processed)
    y_test_pred = regressor.predict(X_test_processed)

    # Evaluation
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)

    train_rmse = root_mean_squared_error(y_train, y_train_pred)
    test_rmse = root_mean_squared_error(y_test, y_test_pred)

    #Actual value and the predicted value
    reg_model_diff = pd.DataFrame({'Actual value': y_test, 'Predicted value': y_test_pred})
    #print(reg_model_diff)

    print("Random Forest Regressor Model Evaluation:")
    print("Training score: {:.2f} %".format(train_r2*100))
    print("Testing score: {:.2f} %".format(test_r2*100))
    print("Training MAE:", round(train_mae,2))
    print("Testing MAE:", round(test_mae,2))
    print("Training MSE:", round(train_mse,2))
    print("Testing MSE:", round(test_mse,2))
    print("Training RMSE:", round(train_rmse,2))
    print("Testing RMSE:", round(test_rmse,2))

    return reg_model_diff

# Call evaluate_model
reg_model_diff = evaluate_model(regressor, X_train_processed, y_train, X_test_processed, y_test)