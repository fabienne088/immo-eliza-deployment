# Import libraries
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import pickle
import gzip

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
with gzip.open('./model/preprocessor.pkl', 'rb') as f:
    preprocessor_loaded = pickle.load(f)

# Preprocess test data using the preprocessor object
data_processed = preprocessor_loaded.transform(new_data_df)

# Load the RF model from the file
with gzip.open('./model/random_forest_regressor.pkl', 'rb') as f:
    loaded_regressor = pickle.load(f)
#print(type(loaded_regressor))

y_new_pred = loaded_regressor.predict(data_processed)
print(f"The price of a new house will be: € {y_new_pred[0]:.2f}")
