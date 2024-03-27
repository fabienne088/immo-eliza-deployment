import streamlit as st
import gzip
import pickle

with gzip.open(r"model\random_forest_regressor.pkl", 'rb') as f:
    model = pickle.load(f)

st.title("House price prediction app")
"""The features are: province, zip_code, total_area_sqm, surface_land_sqm, 
nbr_bedrooms, equipped_kitchen, fl_furnished, fl_open_fire, fl_terrace, terrace_sqm, 
fl_garden, garden_sqm, fl_swimming_pool, state_building, 
primary_energy_consumption_sqm, epc, heating_type, fl_double_glazing.
    """

province = st.selectbox(
   "**Province**",
   ('Antwerp', 'East Flanders', 'Brussels', 'Walloon Brabant', 'Flemish Brabant',
 'Liège', 'West Flanders', 'Hainaut', 'Luxembourg', 'Limburg', 'Namur'),
   index=None,
   placeholder="Select province ...")
st.write('You selected:', province)

 
zip_code = st.number_input(
    '**zip-code**', min_value=612, max_value=9992, 
    value=None, placeholder="Input zip-code")
st.write('The zip-code is:', zip_code)

 
total_area_sqm = st.slider(
    '**total_area_sqm**', 
    min_value=0.0, max_value=15348.0)
st.write(f"The total area is: {total_area_sqm}","sqm")


surface_land_sqm = st.slider(
    '**surface_land_sqm**', 
    min_value=0.0, max_value=950774.0)
st.write(f"The surface land is: {surface_land_sqm}","sqm")

#nbr_bedrooms = 

equipped_kitchen = st.selectbox(
    "**Kitchen-type**", ('INSTALLED','NOT_INSTALLED', 'SEMI_EQUIPPED', 'HYPER_EQUIPPED',
    'USA_HYPER_EQUIPPED', 'USA_INSTALLED', 'USA_UNINSTALLED', 'USA_SEMI_EQUIPPED'),
    index=None,
    placeholder="Select kitchen-type ...")
st.write('You selected:', equipped_kitchen)

#fl_furnished = 
pclass = st.checkbox("furnished")
#fl_open_fire =
#fl_terrace = 

terrace_sqm = st.slider(
    '**terrace_sqm**', 
    min_value=0.0, max_value=3466.0)
st.write(f"The terrace is: {terrace_sqm}","sqm")

#fl_garden = 

garden_sqm = st.slider(
    '**garden_sqm**', 
    min_value=0.0, max_value=150000.0)
st.write(f"The garden is: {garden_sqm}","sqm")

#fl_swimming_pool = 

state_building = st.selectbox(
    "**state_building**", ('TO_RENOVATE', 'GOOD', 'JUST_RENOVATED', 'TO_BE_DONE_UP', 'AS_NEW'),
    index=None,
    placeholder="Select state_building ...")
st.write('You selected:', equipped_kitchen)

primary_energy_consumption_sqm = st.slider(
    '**primary_energy_consumption_sqm**', 
    min_value=0.0, max_value=150000.0)
st.write(f"The primary_energy_consumption_sqm is: {primary_energy_consumption_sqm}","sqm")

#epc = 'A' 'B' 'D' 'F' 'E' 'A+' 'G' 'C' 'A++'
epc= st.selectbox(
    "**EPC**", ('A++','A+', 'A', 'B', 'C', 'D', 'E', 'F', 'G'),
    index=None, placeholder="Select epc type ...")
st.write('You selected:', epc)

#heating_type = 'GAS' nan 'FUELOIL' 'PELLET' 'ELECTRIC' 'CARBON' 'WOOD' 'SOLAR'
heating_type= st.selectbox(
    "**heating_type**", ('GAS','FUELOIL', 'PELLET', 'ELECTRIC', 'CARBON', 'WOOD', 'SOLAR'),
    index=None, placeholder="Select heating type ...")
st.write('You selected:', heating_type)

#fl_double_glazing = 



st.subheader("Streamlit Checkbox basic Examples")
st.checkbox('Unchecked')
st.checkbox('Checked', value=True, key="DefaultChecked")
st.checkbox('Default Disabled', value=False, key="dd", disabled=True)
st.checkbox('Checked Disabled', value=True, key='cd', disabled=True)

st.subheader('Read Checkbox status (checked/unchecked) Examples')
isChecked = st.checkbox('One', key='one')
st.write('Value of isChecked: '+str(isChecked))
if isChecked:
    st.write('CHECKED')
else:
    st.write('UNCHECKED')
    
st.subheader('Basic Examples Done')