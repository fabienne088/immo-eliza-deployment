# Immo-Eliza: deployment


[![forthebadge made-with-python](https://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)

![forthebadge built-with-streamlit](src/built-with-streamlit-🎈.svg)

## 🧐 Description
In the  Immo-Eliza-ML project we build a performant machine learning model to predict prices of real estate proporties in Belgium. In the deployment project we will now create an API and  build a small web application.

![house_logo](src/the-logo-of-home-housing-residents-real-estate-with-a-concept-that-presents-rural-nature-with-a-touch-of-leaves-and-sunflowers-vector.jpg)



## 📦 Repo structure

```md
├── data\
│   ├── cleaned_properties.csv
│   └── properties.csv
│
├── drafts\
│   ├── 1. cleaning.ipynb
│   ├── 2. preprocessing.ipynb
│   ├── 3. model_training_test.ipynb
│   ├── 4. SimpleLinReg.ipynb
│   ├── 5. MultipleLinReg.ipynb
│   ├── 6. RandomForestReg.ipynb
│   ├── 7. functions.ipynb
│   └── 8. functions_cleaned.ipynb
│
├── in_progress\
│   ├── 1. train.ipynb
│   └── 2. predict.ipynb
│
├── models\
│   ├── MultipleLinReg.ipynb
│   └── RandomForestReg.ipynb
│
├── src\
│   └── .jpg
│
├── .gitignore
├── MODELSCARD.md
├── README.md
└── requirements.txt

```

## 🏁 Getting Started

### 📚 Prerequisities

To run the project, you need to install the required libraries. 

You can click on the badge links to learn more about each library and its specific version used in this project. You can install them manually using pip install <library name> or just running pip install -r requirements.txt.

Install the required libraries:

   - [![python version](https://img.shields.io/badge/python-3.x-blue)](https://python.org)
   - [![Pandas Version](https://img.shields.io/badge/pandas-2.x-green)](https://pandas.pydata.org/)
   - [![NumPy Version](https://img.shields.io/badge/numpy-1.x-orange)](https://numpy.org/)
   - [![Matplotlib Version](https://img.shields.io/badge/Matplotlib-3.x-red)](https://matplotlib.org/)
   - [![Seaborn Version](https://img.shields.io/badge/seaborn-0.x-yellow)](https://seaborn.pydata.org/)
   - [![sklearn Version](https://img.shields.io/badge/sklearn-0.x-grey)](https://scikit-learn.org/stable/)
   - ![Static Badge](https://img.shields.io/badge/Streamlit-1.x-%23ff4b4b?color=%23ff4b4b)






### ⚙️ Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/fabienne088/immo-eliza-deployment.git
    ```

2. Navigate to the project directory:
    ```bash
    cd immo-elize-deployment
    ```

3. You're all set! You can now explore the  notebooks in the `drafts` and `models` directories and work with the data in the `data` directory. Enjoy!

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

3. **Explore drafts Notebooks**:
    - The `drafts` directory contains Jupyter notebooks (`*.ipynb`) where you can see step by step cleaning, splitting and preprocessing of the data. Inluding try-outs of model training and evalution. Open these notebooks in Jupyter Notebook or JupyterLab to view the regression analyses, model trainings and results.

4. **Access models**:
    - The `models` directory contains a LinearRegression  and a RandomForestRegression model. Including training, prediction and evaluation.

5. **Work with Data**:
    - The `data` directory contains the dataset used for preprocessing. You can find both original and clean versions of the dataset. Explore the data files to understand their structure and contents.

## ⏱️ Timeline
This project took form in five days.

## 📌 Personal Situation
This project was made as part of the AI Bootcamp at BeCode.org.
