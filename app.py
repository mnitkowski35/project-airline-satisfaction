import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler

# Reading in data

df = pd.read_csv('data/new_train.csv')

# Page title and icon tab
st.set_page_config(page_title = "Airline Passenger Satisfaction", page_icon = ':airplane:')
# Sidebar
st.sidebar.header("Navigate to other pages here!")
page = st.sidebar.selectbox("Select a Page",['Home','Data Overview','Exploratory Data Analysis','Predictive Modeling','Conclusion'])
st.sidebar.divider()
st.sidebar.write("I recommend reading the pages in the order in which they appear on the dropdown menu to gain a more comprehensive understanding of the application.")
st.sidebar.divider()
st.sidebar.write("A special thanks to CodingTemple Data Analytics instructor Katie Sylvia and the rest of the team at CodingTemple for their guidance in building this application.")

# HOMEPAGE
if page == 'Home':
    st.title("Welcome to My Airline Passenger Satisfaction Analysis!")
    st.subheader("Presented by CodingTemple student Matthew Nitkowski")
    st.write("In 2022, airlines in the United States alone carried 853 million passengers. Some had wonderful experiences, and others had terrible experiences. This application is intended to determine what factors cause passengers to feel these ways and how we can use this data to predict passenger satisfaction in the future with the intention that airlines know what factors to focus on in order to improve customer satisfaction. ")
    col1, col2 = st.columns(2)
    with col1:
        st.image("https://static.politico.com/c5/06/afdf202941e599095b821e3e432c/airline-delays-32906.jpg")
        st.caption("Passengers dealing with cancelled flights via airlines Spirit and Allegiant on the week before the Fourth of July")
    with col2:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/2/21/Hartsfield-Jackson_Atlanta_International_Airport_%287039222923%29.jpg/800px-Hartsfield-Jackson_Atlanta_International_Airport_%287039222923%29.jpg")
        st.caption("One of the six domestic terminals at Atlanta's Harsfield-Jackson International Airport, one of the busiest airports in the world")
    st.divider()
    st.subheader("This application's data has been sourced from the dataset website Kaggle. You can find the original source of the data by clicking the box below.")
    st.link_button("Click here to access the original dataset",'https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction/data')

# DATA OVERVIEW 
if page == 'Data Overview':
    st.title("An Intro to the Dataset :bar_chart:")
    st.write("Below is an introduction to the dataset provided from Kaggle. Click on the checkboxes to learn more!")
    st.divider()
    if st.checkbox("Dataframe and Shape"):
        st.write(f"Below is dataframe the Kaggle dataset containing a few modifications I made to the original dataset. There are {df.shape[0]} passengers who are being surveyed and {df.shape[1]} categories in which these passengers are being surveyed.")
        st.dataframe(df)
    if st.checkbox("Column List"):
         st.code(f"Columns: {df.columns.tolist()}")
         st.write('Some of the columns below may look a little odd:')
         list = ['Satisfaction', 'Gender','Customer Type','Type of Travel','Class']
         s = ''
         for category in list:
             s += '- ' + category + "\n"
         st.markdown(s)
         st.write("This is because these are categorical variables that needed to be converted to numbers in order to build prediction models, which we will talk about later.")
    if st.checkbox("Descriptive Statistics"):
        st.write("Below are some descriptive statistics regarding each column in the dataset. Feel free to take a look!")
        st.write(df.describe())

# EDA
if page == 'Exploratory Data Analysis':
    st.title(":mag_right: Time to Analyze the Data! :mag:")
    st.write("On this page, we will be analyzing the individual relationships between any variables of your choosing with three charts- histograms, scatterplots, and boxplots.")
    cols = df.select_dtypes(include = 'number').columns.tolist()
    st.markdown("**Keep in mind that some charts may look a little odd as some categorical variables have been converted to numerical variables.**")
    eda_type = st.multiselect("Choose a visualization you are interested in exploring. You can select more than one at a time!",['Histograms','Box Plots','Scatterplots'])
    if 'Histograms' in eda_type:
        st.subheader("Histograms")
        st.markdown("A **histogram** is a graph that shows the frequency of numerical data.")
        histogram_col = st.selectbox("Choose a column for your histogram!",cols,index = None)
        if histogram_col:
            st.plotly_chart(px.histogram(df, x = histogram_col, title = f"Distribution of {histogram_col.title()}"))
            if histogram_col == 'Class':
                st.write("The class column has been separated into three values, where a number represents the class. Business class is 0, Economy class is 1, and economy plus class is 2.")
            if histogram_col == 'Gender_Male':
                st.write("The gender categories have been split into 0s and 1s. Depending on the gender, the value 1 represents the count of the gender you are viewing.")
            if histogram_col == 'Gender_Female':
                st.write("The gender categories have been split into 0s and 1s. Depending on the gender, the value 1 represents the count of the gender you are viewing.")
            if histogram_col == 'Type of Travel_Personal Travel':
                st.write("The travel categories have been split into 0s and 1s. The value 1 represents the count of the type of travel you are viewing.")
            if histogram_col == 'Type of Travel_Business travel':
                st.write("The travel categories have been split into 0s and 1s. The value 1 represents the count of the type of travel you are viewing.")
            if histogram_col == 'Customer Type_Loyal Customer':
                st.write("The customer type categories have been split into 0s and 1s. The value 1 represents the count of the type of customer you are viewing.")
            if histogram_col == 'Customer Type_disloyal Customer':
                st.write("The customer type categories have been split into 0s and 1s. The value 1 represents the count of the type of customer you are viewing.")
    if 'Box Plots' in eda_type:
        st.subheader("Box Plots")
        st.markdown("A **box plot**, also known as a box and whisker diagram, is a graph summarizing a set of data. The shape of the box plot shows how data is distributed and also includes outliers.")
        boxplot_col = st.selectbox("Select a column for your box plot!", cols, index = None)
        if boxplot_col:
            st.plotly_chart(px.box(df,x = boxplot_col, title = f"Distribution of {boxplot_col.title()}"))
    if 'Scatterplots' in eda_type:
        st.subheader("Scatterplots")
        st.markdown("A **scatterplot** provides a visual means to test the strength of a relationship between two variables. It plots the point where the specified x and y value for each indvidual passenger meets, and creates a **line of best fit** to predict values.")
        selected_col_x = st.selectbox("Select x-axis variables:", cols, index = None)
        selected_col_y = st.selectbox("Select y-axis variables:", cols, index = None)
        if selected_col_x and selected_col_y:
            st.plotly_chart(px.scatter(df, x= selected_col_x, y = selected_col_y, title = f"Relationship Between {selected_col_x.title()} and {selected_col_y.title()} "))
    
# MODELING
if page == 'Predictive Modeling':
    st.title("Time to Model! :gear:")
    st.markdown("**Predictive Modeling** is the concept of using models to predict what our target variable is going to be based on data we are given. In this application, we are attempting to predict what factors cause differences in satisfaction.")
    st.write("There are many different type of machine learning predictive models we can use. On this application, we will be using three models.")
    st.divider()
    st.subheader("But how do we know if our model is any good??")
    st.markdown("A **baseline model** is a simple prediction model in which we will compare the results from our other prediction models. For this baseline model, we will use the mean of each category to predict the value.")
    st.write(f"The mean satisfaction (on a scale of 0 to 1) is {round(df['satisfaction'].mean(),3)}. This means our model needs to predict higher than .433 to be worth using.")
    st.divider()
    features = ['Online boarding','Type of Travel_Business travel','Inflight entertainment','Seat comfort','On-board service','Leg room service','Cleanliness','Flight Distance',
           'Checkin service','Food and drink','Class','Type of Travel_Personal Travel','Customer Type_disloyal Customer','Baggage handling','Inflight wifi service','Inflight service',
           'Customer Type_Loyal Customer']
    st.write("Below are a list of features I chose to include in my prediction models. These features were chosen due to their correlation with our target variable, satisfaction.")
    s = ''
    for category in features:
         s += '- ' + category + "\n"
    st.markdown(s)
    # Setting up X and y and train test split
    X = df[features]
    y = df['satisfaction']
    X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42)
    # Model Selection
    st.subheader("Select a model below. You may only select one at a time.")
    model_option = st.selectbox("Choose a model!", ['Linear Regression','RandomForest','KNN'], index = None)
    if model_option:
        if model_option == 'Linear Regression':
            st.write("Linear regression predicts the value of a variable based on the value of another variable.")
            model= LinearRegression()
        elif model_option == 'RandomForest':
            st.write("Random Forest is a commonly-used machine learning algorithm that combines the output of multiple decision trees to reach a single result.")
            model= RandomForestClassifier()
        elif model_option == 'KNN':
            st.write("KNN, also known as K-Nearest-Neighbors, is a model that predicts the value of a variable based on its nearest neighbors, which is the value k.")
            st.write("K needs to be an odd number so we do not end up with a tie if there is a variation in nearest neighbors.In KNN tests, the default value for k is 5.")
            k_value = st.slider("Select the number of k:",min_value = 1, max_value = 29, step = 2, value= 5)
            
            model = KNeighborsClassifier(n_neighbors = k_value, n_jobs = -1)
        if st.button("Let's check out our results! (May take a bit to load)"):
            model.fit(X_train,y_train)
        # Displaying results
            st.subheader(f"{model} Evaluation:")
            st.write("Below are two percentages: training accuracy and testing accuracy. The training data includes our target variable, satisfaction, while our testing data does not.")
            st.text(f"Training Accuracy: {round(model.score(X_train,y_train),3)}")
            st.text(f"Testing Accuracy: {round(model.score(X_test,y_test), 3)}")
    
# RESULTS
if page == 'Conclusion':
    st.title("Final Results and Conclusion: What Did We Learn? :pencil2:")
    st.subheader("We finally made it to our conclusion! But what did we learn from our predictive models?")
    st.markdown("Remember, we want our models to achieve a better score than .433, or 43.3 percent.")
    st.image("https://media.licdn.com/dms/image/C4D12AQEYUpGT_USmeQ/article-cover_image-shrink_720_1280/0/1602352478948?e=2147483647&v=beta&t=U5F8DKMeVWCOs0MvCkzMeSKom8BrCGZJORjtMLJn2JY")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.header("Linear Regression Results")
        st.subheader("Testing Accuracy: 54%")
        st.write("While better than our baseline model of .433, 54 percent is nowhere near the accuracy we would like it to be. I do not recommend using this model to predict airline satisfaction. Linear regression models are best for predicting numerical values, not categorical ones.")
    with col2:
        st.header("RandomForest Results")
        st.subheader("Testing Accuracy: 95.9%")
        st.write("Wow! RandomForest had an accuracy of almost 96 percent! This is likely due to the structure of RandomForest using multiple decision trees to predict a single result. We get to test a lot of options and use those options to come to a single result. Random Forest is a great predicition tool and I would recommend using it.")
    with col3:
        st.header("K-Nearest Neighbors Results")
        st.subheader("Testing Accuracy: 84-86%")
        st.write("The testing accuracy for KNN is dependent on what you choose for the value of K, but the percentages are very similar, hovering around 84 to 86 percent. While not as accurate as Random Forest, it is still a very solid model and can predict the passenger's satisfaction pretty well.")
    st.divider()
    st.subheader("Conclusion")
    st.write("The variables that had the strongest correlation, either positive or negative, to satisfaction were online boarding(0.5), class(-0.45), and type of travel, either business or personal. Correlation does not equate to causation, but recognizing these correlations and seeing how they affect our predictive models are important. There were other factors such as inflight entertainment, seat comfort, on-board service, and leg room that had a pretty significant impact as well. To all airlines out there, I recommend you consider these features significantly when hoping to satisfy your customers.")
