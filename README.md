# Immo-Eliza: deployment


[![forthebadge made-with-python](https://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)

[![forthebadge built-with-streamlit](src/built-with-streamlit-🎈.svg)](https://streamlit.io/)

## 🧐 Description
In the  Immo-Eliza-ML project we build a performant machine learning model to predict prices of real estate proporties in Belgium. In the deployment project we will now create an API and  build a small web application.

![house_logo](src/the-logo-of-home-housing-residents-real-estate-with-a-concept-that-presents-rural-nature-with-a-touch-of-leaves-and-sunflowers-vector.jpg)



## 📦 Repo structure

```md
├── data\
│   ├── cleaned_properties.csv
│   └── data.ipynb
│
├── model\
│   │
│   ├── __pycache__\
│   │   └── train.cpython-312.pyc
│   │
│   ├── predict.py
│   ├── preprocessor.pkl
│   ├── random_forest_regressor.pkl
│   └── train.py
│
├── src\
│   ├── built-with-streamlit-🎈.svg
│   └── the-logo-of-...jpg
│
├── .gitignore
├── README.md
├── app.py
└── requirements.txt

```

## 🏁 Getting Started

### 📚 Prerequisities

To run the project, you need to install the required libraries. 

You can click on the badge links to learn more about each library and its specific version used in this project. You can install them manually using pip install <library name> or just running pip install -r requirements.txt.

Install the required libraries:

   - [![python version](https://img.shields.io/badge/python-3.x-blue)](https://python.org)
   - [![Pandas version](https://img.shields.io/badge/pandas-2.x-green)](https://pandas.pydata.org/)
   - [![NumPy version](https://img.shields.io/badge/numpy-1.x-orange)](https://numpy.org/)
   - [![matplotlib version](https://img.shields.io/badge/matplotlib-3.x-red)](https://matplotlib.org/)
   - [![Seaborn version](https://img.shields.io/badge/seaborn-0.x-yellow)](https://seaborn.pydata.org/)
   - [![sklearn version](https://img.shields.io/badge/scikit_learn-1.x-%f89938?color=%f89938)](https://scikit-learn.org/stable/)
   - [![Streamlit version](https://img.shields.io/badge/Streamlit-1.x-%23ff4b4b?color=%23ff4b4b)](https://streamlit.io/)

### ⚙️ Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/fabienne088/immo-eliza-deployment.git
    ```

2. Navigate to the project directory:
    ```bash
    cd immo-elize-deployment
    ```

3. You're all set! You can now explore the `data` and `model` directories. You can find the script for the app in `app.py`.<br>
Enjoy!

## 🎈 Usage
To use this repository, follow these steps:

1. **Clone the Repository**: 
    - Clone the repository to your local machine using the following command:
    ```bash
    git clone https://github.com/fabienne088/immo-eliza-deployment.git
    ```

2. **Navigate to the Project Directory**:
    - Once cloned, navigate to the project directory:
    ```bash
    cd immo-elize-deployment
    ```

3. **Explore the data**:
    - The `data` directory contains the dataset used. Explore the data file to understand its structure and contents. You can use the included Jupyter Notebook for it.

4. **Access the model**:
    - The `model` directory contains a train.py that trains the RandomForestRegression model and evaluate it, and a predict.py that predicts the price of a house based on a given dataset.

5. **Explore the script app.py**
    - The `app.py` script contains everything for running the app.

6. **Open the app** 🎉

    [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://immo-eliza-deployment-3pzskfv3wyawvpj4u52pwg.streamlit.app/)

## 🎨 Visuals
Will be added soon.

## 👑 Resources
[Streamlit documentation](https://docs.streamlit.io/)

[Deploy your app](https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app)


## ⏱️ Timeline
This project took form in five days.

## 📌 Personal Situation
This project was made as part of the AI Bootcamp at BeCode.org.

## 🔧 Maintainers
Connect with me on [LinkedIn](https://www.linkedin.com/in/fabienne-th%C3%BCer-56a8a0a?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base_contact_details%3BGVOLSNIkQnaKEDrsWD%2BY6w%3D%3D).
