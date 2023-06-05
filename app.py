import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load the data
@st.cache_data  # Add caching for better performance
def load_data():
    data = pd.read_csv('C:\\Users\\chuan\\Downloads\\Maternal Health Risk Data Set.csv.xls')
    return data

data = load_data()

# Label encode the target variable
le = LabelEncoder()
data['RiskLevel'] = le.fit_transform(data['RiskLevel'])

# Define the column order
feature_names = ['Age', 'BS', 'BodyTemp', 'DiastolicBP', 'HeartRate', 'SystolicBP']

# Create sidebar
st.sidebar.subheader('Data Exploration')
section = st.sidebar.selectbox('Section', ('Data Rows', 'RiskLevel Counts', 'Data Description', 'RiskLevel Prediction'))

# Display selected section
if section == 'Data Rows':
    st.subheader("First 10 rows of the data:")
    st.write(data.head(10))
elif section == 'RiskLevel Counts':
    st.subheader("Counts of RiskLevel for each Age:")
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Age', hue='RiskLevel', data=data)
    plt.title('Counts of RiskLevel for each Age')
    plt.xlabel('Age')
    plt.ylabel('Count')
    st.pyplot(plt)
elif section == 'Data Description':
    st.subheader("Description of Data:")
    st.write(data.describe().T)
elif section == 'RiskLevel Prediction':
    st.subheader("Predict RiskLevel")
    age = st.number_input("Enter Age", min_value=1, max_value=100, value=30)
    bs = st.number_input("Enter Blood Sugar", min_value=0.0, max_value=300.0, value=100.0)
    body_temp = st.number_input("Enter Body Temperature", min_value=35.0, max_value=42.0, value=37.0)
    diastolic_bp = st.number_input("Enter Diastolic Blood Pressure", min_value=0, max_value=200, value=80)
    heart_rate = st.number_input("Enter Heart Rate", min_value=0, max_value=200, value=70)
    systolic_bp = st.number_input("Enter Systolic Blood Pressure", min_value=0, max_value=300, value=120)

    input_data = pd.DataFrame({
        'Age': [age],
        'BS': [bs],
        'BodyTemp': [body_temp],
        'DiastolicBP': [diastolic_bp],
        'HeartRate': [heart_rate],
        'SystolicBP': [systolic_bp]
    }, columns=feature_names)

    # Create a classification model
    X = data[feature_names]
    y = data['RiskLevel']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(max_depth=2, random_state=0)
    model.fit(X_train, y_train)

    prediction = model.predict(input_data)  # Make a prediction using the trained model

    predicted_label = le.inverse_transform(prediction)[0]
    st.write("Predicted RiskLevel:", predicted_label)
