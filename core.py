# Dependencies.
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import streamlit as st
import seaborn as sns
from matplotlib import pyplot as plt
#import matplotlib.pyplot as plt

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

    licit_col, illicit_col = st.columns((1, 1))

    licit_col.subheader('Licit Transactions Description')
    licit = licit.drop('Time', axis = 1)
    licit_col.write(licit.describe())

    illicit_col.subheader('Illicit Transactions Description')
    illicit = illicit.drop('Time', axis = 1)
    illicit_col.write(illicit.describe())

elif navigation == 'Visualization':

    # Separating data classes.
    licit = data[data.Class == 0]
    illicit = data[data.Class == 1]

    # Pairplot.
    st.subheader("Pairplots")
    #licit = licit.sample(n = 492)
    #sample_data = pd.concat([licit, illicit])
    #fig = sns.pairplot(sample_data[features[1:]], hue = 'Class')
    #plots_col1.pyplot(fig)
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = 205977227
    image = Image.open('pairplot.png')
    st.image(image)

    # Plots.
    plots_col1, plots_col2 = st.columns((1, 1))
   
    plots_col1.subheader("Heatmap: Linear Correlations")
    plots_col2.subheader("Heatmap: Non-linear Correlations")
    
    # Linear correlation using heatmap.
    #fig = plt.figure(figsize = (35, 20))
    #sns.heatmap(data[features[1:-1]].corr(), annot = True, cmap = "YlGnBu")
    #plots_col2.pyplot(fig)
    image = Image.open('linear_correlations_heatmap.png')
    plots_col1.image(image)

    # Non-linear Correlation testing with PPS.
    import ppscore as pps
    '''fig = plt.figure(figsize = (35, 20))
    ppsMatrix = pps.matrix(data[features[1:-1]]).pivot(columns = 'x', index = 'y',  values = 'ppscore')
    sns.heatmap(ppsMatrix, annot = True)
    plots_col3.pyplot(fig)'''
    from PIL import Image
    image = Image.open('ppscore_heatmap.png')
    plots_col2.image(image)
    
