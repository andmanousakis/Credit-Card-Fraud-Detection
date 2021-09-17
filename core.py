# Dependencies.
import numpy as np
import pandas as pd
#from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LogisticRegression
#from sklearn.metrics import accuracy_score
#from sklearn.preprocessing import StandardScaler
import streamlit as st

# Importing dataset.
data = pd.read_csv('data.csv')

# Features.
features = data.columns

##
##
##
## StreamLit Application.
##
##
##

def separator(area):

    line = '---'
    if area == 'sidebar':
        st.sidebar.markdown(line)
    elif area == 'main':
        st.markdown(line)

st.set_page_config(layout = "wide")

st.title("Credit Card Fraud Detection Dashboard")
separator('main')

st.sidebar.subheader('Navigation')
navigation = st.sidebar.radio(" ", ("Data Description & Cleaning", "Visualization", "Classification"))

if navigation == 'Data Description & Cleaning':

    # Sidebar.
    separator('sidebar')
    st.sidebar.write('Dataset Shape:', data.shape)
    #st.sidebar.write('Nr. of classes:', set(y_labels))

    # Data.
    st.subheader('Data')
    st.write(data.head(10))

    # Data Description.
    data_description_col1, data_description_col2 = st.columns((1, 1))
    data_description_col1.subheader('Data Description'); 
    data_description_col2.subheader('Data Info')
    data_description_col1.write(data.describe())

    # Data info.     
    dataInfo = {el:'0' for el in features}
    rows = ['Non-Null Count', 'Missing Values', 'dtype']
    dataInfoDF = pd.DataFrame(dataInfo, index = rows, columns = features)
    dataInfoDF.loc['Non-Null Count'] = str(data.value_counts(dropna = True).sum())
    dtypes = data[features].dtypes
    dataInfoDF.loc['dtype'] = [str(i) for i in dtypes]

    data_description_col2.write(dataInfoDF.head(10))
    missing_values = []
    for f in features:

        missing_values.append(data[f].isnull().sum())
    dataInfoDF.loc['Missing Values'] = str(missing_values)

    # Distribution of licit & illicit transactions.
    classes = data['Class'].value_counts()
    classesDF = pd.DataFrame(classes.to_numpy(), columns = ['Class'], index = ['Licit (0)', 'Illicit (1)'])
    classesDF['Percent'] = classesDF['Class'].apply(lambda x : 100*float(x) / data.shape[0])
    data_description_col2.write(classesDF)

    # Separating data classes.
    licit = data[data.Class == 0]
    illicit = data[data.Class == 1]

    licit_col, illicit_col, comparison_col = st.columns((1, 1, 3))

    licit_col.subheader('Licit Transactions')
    licit_col.write(licit.Amount.describe())

    illicit_col.subheader('Illicit Transactions')
    illicit_col.write(illicit.Amount.describe())

    comparison_col.subheader('Comparison')
    comparison_col.write(data.groupby('Class').mean())

elif navigation == 'Visualization':

    # Plotting.
    pass
