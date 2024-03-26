import streamlit as st
import gzip
import pickle

with gzip.open(r"model\random_forest_regressor.pkl", 'rb') as f:
    model = pickle.load(f)

st.title("House price prediction app")
"""The features are: province, zip_code, total_area_sqm, surface_land_sqm, 
nbr_bedrooms, equipped_kitchen, fl_furnished, fl_terrace, terrace_sqm, fl_garden, garden_sqm, fl_swimming_pool, 
state_building, primary_energy_consumption_sqm, epc, heating_type, fl_double_glazing.
    """

province = st.selectbox(
   "Province",
   ('Antwerp', 'East Flanders', 'Brussels', 'Walloon Brabant', 'Flemish Brabant',
 'Liège', 'West Flanders', 'Hainaut', 'Luxembourg', 'Limburg', 'Namur'),
   index=None,
   placeholder="Select province...",
)
st.write('You selected:', province)

#zip_code = 
#total_area_sqm = 
#surface_land_sqm = 
#nbr_bedrooms = 

equipped_kitchen = st.selectbox(
    "Kitchen-type", ('INSTALLED','NOT_INSTALLED', 'SEMI_EQUIPPED', 'HYPER_EQUIPPED',
    'USA_HYPER_EQUIPPED', 'USA_INSTALLED', 'USA_UNINSTALLED', 'USA_SEMI_EQUIPPED'),
    index=None,
    placeholder="Select kitchen-type...",
)


#fl_furnished = 
#fl_terrace = 
#terrace_sqm = 
#fl_garden = 
#garden_sqm = 
#fl_swimming_pool = 
#state_building = 'TO_RENOVATE' 'GOOD' 'JUST_RENOVATED' 'TO_BE_DONE_UP' 'AS_NEW'
#primary_energy_consumption_sqm = 
#epc = 'A' 'B' 'D' 'F' 'E' 'A+' 'G' 'C' 'A++'
#heating_type = 'GAS' nan 'FUELOIL' 'PELLET' 'ELECTRIC' 'CARBON' 'WOOD' 'SOLAR'
#fl_double_glazing = 