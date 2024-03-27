import pandas as pd
import streamlit as st
import gzip
import pickle
from sklearn.model_selection import train_test_split

# Read the csv file
df = pd.read_csv(r"data\cleaned_properties.csv")
df_house = df[(df["property_type"] == "HOUSE") & (df['subproperty_type'] != 'APARTMENT_BLOCK')]

# Name X and y
X = df_house.drop(columns=['price', 'subproperty_type', 'property_type', 'region', 'locality', 'construction_year', 'cadastral_income', 'nbr_frontages', 'fl_floodzone'])
y = df_house['price']

# Split the data into training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)


st.title("House price prediction app")
"""The features are: province, zip_code, total_area_sqm, surface_land_sqm, 
nbr_bedrooms, equipped_kitchen, fl_furnished, fl_open_fire, fl_terrace, terrace_sqm, 
fl_garden, garden_sqm, fl_swimming_pool, state_building, 
primary_energy_consumption_sqm, epc, heating_type, fl_double_glazing.
    """

# Location
st.subheader("Location")

province = st.selectbox("**Province**",
   options=sorted(X['province'].unique()),
   index=None, placeholder="Select a province ...")
st.write('You selected:', province)

zip_code = st.number_input('**Postal code**', 
    min_value = X['zip_code'].min(), max_value=X['zip_code'].max(), 
    value=None, placeholder="Enter a postal code")
st.write('The postal code is:', zip_code)

# Infrastructure
st.divider()
st.subheader("Infrastructure")

surface_land_sqm = st.slider('**Surface land**', 
    min_value=0.0, max_value=X['surface_land_sqm'].max())
st.write(f"The surface land is: {surface_land_sqm}","sqm")

total_area_sqm = st.slider('**Total living area**', 
    min_value=0.0, max_value=X['total_area_sqm'].max())
st.write(f"The total living area is: {total_area_sqm}","sqm")

# Extra infrastructure
st.write('**Extra infrastructure**')

fl_swimming_pool = int(st.checkbox('swimming pool', key='fl_swimming_pool'))
if fl_swimming_pool:
    st.write("There is a swimming pool!")

# Interior
st.divider()
st.subheader("Interior")

nbr_bedrooms = st.number_input('**number of bedrooms**',
    min_value=0, max_value=int(X['nbr_bedrooms'].max()), step=1)
st.write(f"There are {nbr_bedrooms} bedrooms")

# Extra interior elements
st.subheader("Extra interior elements")

furnished_str = st.selectbox("**Furnished**",
   options=["Yes", "No"],
   index=None,  # Set the default index
   placeholder="Select")
# Convert the selected value to a boolean (True or False)
furnished = furnished_str == 'Yes'
fl_furnished = int(furnished)
st.write('You selected:', furnished_str)

open_fire_str = st.selectbox("**Open fire**",
   options=["Yes", "No"], index=None, placeholder="Select")
open_fire = open_fire_str == 'Yes'
fl_open_fire = int(open_fire)
st.write('You selected:', open_fire_str)

equipped_kitchen = st.selectbox("**Kitchen set-up**", 
    options=sorted(X['equipped_kitchen'].dropna().unique()),
    index=None, placeholder="Select set-up ...")
st.write('You selected:', equipped_kitchen)

# Exterior
st.divider()
st.subheader("Exterior")
st.write("Is there a terrace and/or a garden?")

fl_terrace = int(st.checkbox('terrace', key='fl_terrace'))
if fl_terrace:
    st.write('Please provide Terrace surface sqm')
    terrace_sqm = st.slider('**Terrace surface**', 
                min_value=0.0, max_value=X['terrace_sqm'].max())
    st.write(f"The terrace is: {terrace_sqm}","sqm")
else:
    terrace_sqm = 0.00

fl_garden = int(st.checkbox('garden', key='fl_garden'))
if fl_garden:
    st.write('Please provide Garden surface sqm')
    garden_sqm = st.slider('**Garden surface**', 
               min_value=0.0, max_value=X['garden_sqm'].max())
    st.write(f"The garden is: {garden_sqm}","sqm")
else:
    garden_sqm = 0.00


st.divider()
st.subheader("Building condition")

state_building = st.selectbox("**State**", 
    options=sorted(X['state_building'].dropna().unique()),
    index=None, placeholder="Select state_building ...")
st.write('You selected:', state_building)

st.divider()
st.subheader("Energy")

primary_energy_consumption_sqm = st.slider('**Primary energy consumption**', 
    min_value=0.0, max_value=X['primary_energy_consumption_sqm'].max())
st.write(f"The primary energy consumption is: {primary_energy_consumption_sqm}","sqm")

epc= st.selectbox("**Energy class**", 
    options=sorted(X['epc'].dropna().unique()),
    index=None, placeholder="Select class ...")
st.write('You selected:', epc)

heating_type= st.selectbox("**Heating type**", 
    options=sorted(X['heating_type'].dropna().unique()),
    index=None, placeholder="Select heating type ...")
st.write('You selected:', heating_type)

double_glazing_str = st.selectbox("**Double glazing**",
   options=["Yes", "No"],
   index=None, placeholder="Select")
double_glazing = double_glazing_str == 'Yes'
fl_double_glazing = int(double_glazing)
st.write('You selected:', double_glazing_str)


data = {
    'province': province,
    'zip_code': zip_code,
    'total_area_sqm': total_area_sqm ,
    'surface_land_sqm': surface_land_sqm,
    'nbr_bedrooms': nbr_bedrooms,
    'equipped_kitchen': equipped_kitchen,
    'fl_furnished': fl_furnished,
    'fl_open_fire': fl_open_fire,
    'fl_terrace': fl_terrace,
    'terrace_sqm': terrace_sqm,
    'fl_garden': fl_garden,
    'garden_sqm': garden_sqm,
    'fl_swimming_pool': fl_swimming_pool,
    'state_building': state_building,
    'primary_energy_consumption_sqm': primary_energy_consumption_sqm,
    'epc': epc,
    'heating_type': heating_type,
    'fl_double_glazing': fl_double_glazing
}

# Define row

#row = np.array([province, zip_code, total_area_sqm, surface_land_sqm, 
#nbr_bedrooms, equipped_kitchen, fl_furnished, fl_open_fire, fl_terrace, terrace_sqm, 
#fl_garden, garden_sqm, fl_swimming_pool, state_building, 
#primary_energy_consumption_sqm, epc, heating_type, fl_double_glazing]) 

# Define columns
# columns = X.columns.tolist()


# Create a new DataFrame
new_data_df = pd.DataFrame(data, index=[0])

# Load the preprocessor
with gzip.open(r"model\preprocessor.pkl", 'rb') as f:
    preprocessor = pickle.load(f)
# Import the model
with gzip.open(r"model\random_forest_regressor.pkl", 'rb') as f:
    model = pickle.load(f)

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

    return new_data_processed

# Preprocess test data using the preprocessor object
new_data_processed = preprocess_data_for_test(new_data_df, preprocessor)

prediction = model.predict(new_data_processed)


#def predict():



#trigger = st.button('Predict', on_click=predict)