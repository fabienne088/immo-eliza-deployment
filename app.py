import pandas as pd
import numpy as np
import streamlit as st
import gzip
import pickle
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor


# Read the csv file
df = pd.read_csv("./data/cleaned_properties.csv")
df_house = df[(df["property_type"] == "HOUSE") & (df['subproperty_type'] != 'APARTMENT_BLOCK')]

# Name X and y
X = df_house.drop(columns=['price', 'subproperty_type', 'property_type', 'region', 'locality', 'construction_year', 'cadastral_income', 'nbr_frontages', 'fl_floodzone'])
y = df_house['price']

# Split the data into training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)


st.title(":green[House] price :blue[prediction] app")

st.image('./src/the-logo-of-home-housing-residents-real-estate-with-a-concept-that-presents-rural-nature-with-a-touch-of-leaves-and-sunflowers-vector.jpg')

st.write("This app will predict the price of your new house. :house:")

with st.sidebar:
    st.markdown("**The features :house_with_garden: are:**  \n province, "
    " postal code, total area (sqm), surface land (sqm),"
    " number of bedrooms, equipped kitchen, furnished, open fire,"
    " terrace, terrace (sqm), garden, garden (sqm), swimming pool,"
    " building condition, primary energy consumption (sqm), epc,"
    " heating type, and double glazing.")    
    

# Location
st.subheader("Location :round_pushpin:")

province = st.selectbox("**Province**",
   options=sorted(X['province'].unique()),
   index=None, placeholder="Select a province ...")
st.write('You selected:', province)

zip_code = st.number_input('**Postal code**', 
    min_value = int(X['zip_code'].min()), max_value=int(X['zip_code'].max()), 
    placeholder="Enter a postal code")
st.write('The postal code is:', zip_code)

# Infrastructure
st.divider()
st.subheader("Infrastructure :straight_ruler:")

surface_land_sqm = st.slider('**Surface Land (sqm)**', 
    min_value=0.0, max_value=X['surface_land_sqm'].quantile(q=0.7), step=1.00)
st.write(f"The surface land is: {surface_land_sqm}","sqm")

total_area_sqm = st.slider('**Total Living Area (sqm)**', 
    min_value=0.0, max_value=X['total_area_sqm'].quantile(q=0.7), step=1.00)
st.write(f"The total living area is: {total_area_sqm}","sqm")

# Extra infrastructure
st.write('**Extra infrastructure**')

fl_swimming_pool = int(st.checkbox('swimming pool :ocean:', key='fl_swimming_pool'))
if fl_swimming_pool:
    st.write("There is a swimming pool! :swimmer:")

# Interior
st.divider()
st.subheader("Interior :frame_with_picture:")

nbr_bedrooms = st.number_input('**number of bedrooms** :bed:',
    min_value=0, max_value=int(X['nbr_bedrooms'].quantile(q=0.9)), step=1)
st.write(f"There are {nbr_bedrooms} bedrooms")

# Extra interior elements
st.subheader("Extra interior elements")

furnished_str = st.selectbox("**Furnished** :couch_and_lamp:",
   options=["Yes", "No"],
   index=None,  # Set the default index
   placeholder="Select")
# Convert the selected value to a boolean (True or False)
furnished = furnished_str == 'Yes'
fl_furnished = int(furnished)
st.write('You selected:', furnished_str)

open_fire_str = st.selectbox("**Open fire** :fire:",
   options=["Yes", "No"], index=None, placeholder="Select")
open_fire = open_fire_str == 'Yes'
fl_open_fire = int(open_fire)
st.write('You selected:', open_fire_str)

equipped_kitchen = st.selectbox("**Kitchen set-up** :fork_and_knife: ", 
    options=sorted(X['equipped_kitchen'].dropna().unique()),
    index=None, placeholder="Select set-up ...")
st.write('You selected:', equipped_kitchen)

# Exterior
st.divider()
st.subheader("Exterior")
st.write("Is there a terrace and/or a garden? :house_with_garden:")

fl_terrace = int(st.checkbox('terrace', key='fl_terrace'))
if fl_terrace:
    st.write('Please provide Terrace surface (sqm)')
    terrace_sqm = st.slider('**Terrace surface (sqm)**', 
                min_value=0.0, max_value=X['terrace_sqm'].quantile(q=0.8), step=0.10)
    st.write(f"The terrace is: {terrace_sqm}","sqm")
else:
    terrace_sqm = 0.00

fl_garden = int(st.checkbox('garden', key='fl_garden'))
if fl_garden:
    st.write('Please provide Garden surface (sqm)')
    garden_sqm = st.slider('**Garden surface (sqm)**', 
               min_value=0.0, max_value=X['garden_sqm'].quantile(q=0.8), step=0.10)
    st.write(f"The garden is: {garden_sqm}","sqm")
else:
    garden_sqm = 0.00

st.divider()
st.subheader("Building condition :toolbox:")

state_building = st.selectbox("**State**", 
    options=sorted(X['state_building'].dropna().unique()),
    index=None, placeholder="Select condition ...")
st.write('You selected:', state_building)

st.divider()
st.subheader("Energy")
st.image('./src/EPCwoningen_Y18-2309_VO_Label_06-01_Aplus_ivrxha.jpg')

primary_energy_consumption_sqm = st.slider('**Primary energy consumption (sqm)**', 
    min_value=0.0, max_value=X['primary_energy_consumption_sqm'].quantile(q=0.7), step=1.00)
st.write(f"The primary energy consumption is: {primary_energy_consumption_sqm}","sqm")

epc= st.selectbox("**Energy class**", 
    options=sorted(X['epc'].dropna().unique()),
    index=None, placeholder="Select class ...")
st.write('You selected:', epc)

heating_type= st.selectbox("**Heating type :sunny:**", 
    options=sorted(X['heating_type'].dropna().unique()),
    index=None, placeholder="Select heating type ...")
st.write('You selected:', heating_type)

double_glazing_str = st.selectbox("**Double glazing** :window: ",
   options=["Yes", "No"],
   index=None, placeholder="Select")
double_glazing = double_glazing_str == 'Yes'
fl_double_glazing = int(double_glazing)
st.write('You selected:', double_glazing_str)

def predict(data):

    # Load the preprocessor
    with gzip.open('./model/preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
    
    data_processed = preprocessor.transform(data)
    
    # Import the model
    with gzip.open('./model/random_forest_regressor.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Predict the price of the new house
    prediction = model.predict(data_processed)

    return prediction.tolist()[0]

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

# Create a new DataFrame
new_data_df = pd.DataFrame(data, index=[0])
print(new_data_df.info())

# Perform predictions when button is clicked
if st.button('Predict'):
    predicted_price = predict(new_data_df)
    st.success(f"The price of the new house will be: € {predicted_price:.2f}")

