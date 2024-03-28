# So we can save the encoding object we used in the X-train into a pickle file, as well as save the scaling object into a pickle file.
# And when we are going to make a new prediction with new data:
# We load the encoding prickle file, use it to transform the new data, then do the same with the scaler, and then with the model.

# Import libraries
import pandas as pd
import numpy as np
import csv
# from train import *
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import pickle
import gzip
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error

# Sample data for the new DataFrame
data = {
    'province': 'East Flanders',
    'zip_code': 9630,
    'total_area_sqm': 150.0 ,
    'surface_land_sqm': 200.0,
    'nbr_bedrooms': 3,
    'equipped_kitchen': 'HYPER_EQUIPPED',
    'fl_furnished': 1,
    'fl_open_fire': 1,
    'fl_terrace': 1,
    'terrace_sqm': 25.0,
    'fl_garden': 1,
    'garden_sqm': 50.0,
    'fl_swimming_pool': 1,
    'state_building': 'GOOD',
    'primary_energy_consumption_sqm': 150,
    'epc': 'A',
    'heating_type': 'GAS',
    'fl_double_glazing': 1
}

# Create a new DataFrame
new_data_df = pd.DataFrame(data, index=[0])

# Display the new DataFrame
print(new_data_df.head().T)


# Load the preprocessor
with gzip.open(r"model\preprocessor.pkl", 'rb') as f:
    preprocessor_loaded = pickle.load(f)
#print(type(preprocessor_loaded))

# Load the preprocess_data function
from train import preprocess_data_for_test

# Preprocess test data using the preprocessor object
new_data_processed = preprocess_data_for_test(new_data_df, preprocessor_loaded)
# Display the new DataFrame
#print(new_data_processed.head().T)

# Load the RF model from the file
with gzip.open(r"model\random_forest_regressor.pkl", 'rb') as f:
    loaded_regressor = pickle.load(f)
#print(type(loaded_regressor))

y_new_pred = loaded_regressor.predict(new_data_processed)
print(f"The price of a new house will be: € {y_new_pred[0]:.2f}")
