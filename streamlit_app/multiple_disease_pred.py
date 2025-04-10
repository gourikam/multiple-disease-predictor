import streamlit as st
import requests
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import time

# Function to load Lottie animations
def load_lottieurl(url):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

# Page configuration
st.set_page_config(
    page_title="Disease Prediction System",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem !important;
        color: #1E88E5;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem !important;
        color: #0277BD;
    }
    .card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-card {
        background: linear-gradient(to right, #1E88E5, #64B5F6);
        color: white;
        padding: 15px;
        border-radius: 8px;
        text-align: center; 
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .info-text {
        background-color: #e3f2fd;
        border-left: 5px solid #1E88E5;
        padding: 10px;
        border-radius: 5px;
    }
    .stButton button {
        background-color: #1E88E5;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: bold;
        width: 100%;
    }
    .result-positive {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 15px;
        border-radius: 5px;
    }
    .result-negative {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
        padding: 15px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar navigation
with st.sidebar:
    # Add a logo or animation at the top of the sidebar
    lottie_health = load_lottieurl("https://lottie.host/3c99f99a-b0e7-4794-b8ef-13a432cd77ce/BdT39WZsXl.json")
    if lottie_health:
        st_lottie(lottie_health, height=150, key="health_animation")
    else:
        st.image("https://cdn-icons-png.flaticon.com/512/3004/3004458.png", width=150)
    
    st.markdown("### Navigate")
    selected = option_menu(
        'Disease Prediction System',
        ['Home', 'Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction'],
        icons=['house', 'activity', 'heart', 'person'],
        default_index=0
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info("This application uses machine learning to predict disease likelihood based on medical parameters.")

# API URL - make sure this matches your FastAPI server address
API_URL = "http://127.0.0.1:8000"

# Sample data for visualizations (to be used when no real data is available)
def generate_sample_data():
    # Sample data for visualizations
    diabetes_data = {
        'Factor': ['Glucose Level', 'BMI', 'Age', 'Blood Pressure', 'Insulin Level'],
        'Importance': [32, 27, 18, 14, 9]
    }
    
    heart_data = {
        'Factor': ['Age', 'Cholesterol', 'Blood Pressure', 'Max Heart Rate', 'ST Depression'],
        'Importance': [25, 22, 19, 18, 16]
    }
    
    parkinsons_data = {
        'Factor': ['PPE', 'RPDE', 'DFA', 'Spread1', 'Shimmer'],
        'Importance': [30, 25, 20, 15, 10]
    }
    
    return diabetes_data, heart_data, parkinsons_data

diabetes_data, heart_data, parkinsons_data = generate_sample_data()

# Function to create gauge chart for prediction probability
def create_gauge_chart(probability, title):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 24}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#4CAF50'},
                {'range': [50, 75], 'color': '#FFC107'},
                {'range': [75, 100], 'color': '#F44336'}
            ],
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig

# Function to visualize input parameters
def visualize_parameters(input_data, disease_type):
    if disease_type == 'diabetes':
        # Select key parameters
        if len(input_data) >= 8:
            key_params = {
                'Glucose': input_data.get('Glucose', 0),
                'BMI': input_data.get('BMI', 0),
                'Age': input_data.get('Age', 0),
                'BloodPressure': input_data.get('BloodPressure', 0),
                'Insulin': input_data.get('Insulin', 0)
            }
        else:
            return None
    elif disease_type == 'heart':
        # Select key parameters
        if len(input_data) >= 5:
            key_params = {
                'Age': input_data.get('age', 0),
                'Cholesterol': input_data.get('chol', 0),
                'BloodPressure': input_data.get('trestbps', 0),
                'HeartRate': input_data.get('thalach', 0),
                'STDepression': input_data.get('oldpeak', 0)
            }
        else:
            return None
    else:  # Parkinson's
        # For Parkinson's, we'll choose a different visualization due to many parameters
        return None
    
    # Create radar chart for selected parameters
    categories = list(key_params.keys())
    values = list(key_params.values())
    
    # Normalize values for better visualization 
    # (this is simplified and would require actual scaling based on normal ranges in real app)
    normalized_values = []
    for k, v in zip(categories, values):
        if k == 'Age':
            normalized_values.append(min(v / 100, 1))
        elif k == 'Glucose' or k == 'BloodPressure' or k == 'HeartRate':
            normalized_values.append(min(v / 200, 1))
        elif k == 'BMI':
            normalized_values.append(min(v / 50, 1))
        elif k == 'Cholesterol':
            normalized_values.append(min(v / 300, 1))
        elif k == 'Insulin':
            normalized_values.append(min(v / 500, 1))
        elif k == 'STDepression':
            normalized_values.append(min(v / 5, 1))
        else:
            normalized_values.append(min(v / 100, 1))
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=normalized_values,
        theta=categories,
        fill='toself',
        line=dict(color='#1E88E5')
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        title="Parameter Analysis",
        height=400
    )
    return fig

# Home page
if selected == 'Home':
    st.markdown('<h1 class="main-header">🏥 Health Guardian - Disease Prediction System</h1>', unsafe_allow_html=True)
    
    # Animation or illustration
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## About this Application
        This intelligent system uses machine learning algorithms to predict the likelihood of three common diseases:
        
        - **Diabetes**
        - **Heart Disease**
        - **Parkinson's Disease**
        
        Our models are trained on extensive healthcare datasets and provide quick preliminary health assessments. This tool is designed to complement, not replace, professional medical advice.
        """)
        
        st.markdown("### How to use")
        st.markdown("""
        1. Select a disease from the sidebar menu
        2. Enter your health parameters
        3. Click on the prediction button to get your result with visualization
        """)
    
    with col2:
        # Add a health-related animation
        lottie_doctor = load_lottieurl("https://lottie.host/c99d750f-5d11-4b2a-987e-e1c36f5c2bcd/MFBf2Ksfq1.json")
        if lottie_doctor:
            st_lottie(lottie_doctor, height=250, key="doctor_animation")
    
    st.markdown("---")
    
    # Metrics Section
    st.markdown('<h2 class="sub-header">Disease Impact Metrics</h2>', unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    
    with m1:
        st.markdown('<div class="metric-card"><h2>463 million</h2>Adults living with diabetes worldwide</div>', unsafe_allow_html=True)
    
    with m2:
        st.markdown('<div class="metric-card"><h2>17.9 million</h2>Annual deaths from cardiovascular disease</div>', unsafe_allow_html=True)
    
    with m3:
        st.markdown('<div class="metric-card"><h2>10 million</h2>People living with Parkinson\'s disease globally</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Display some information about diseases in columns with visualization
    st.markdown('<h2 class="sub-header">Disease Information</h2>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Diabetes")
        st.write("""
        Diabetes is a chronic disease that occurs when the pancreas is no longer able 
        to make insulin, or when the body cannot make good use of the insulin it produces.
        """)
        
        # Create a bar chart for diabetes risk factors
        fig = px.bar(
            diabetes_data, 
            x='Factor', 
            y='Importance',
            title='Key Factors in Diabetes Prediction',
            color='Importance',
            color_continuous_scale='blues'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Heart Disease")
        st.write("""
        Heart disease describes a range of conditions that affect your heart, 
        including coronary artery disease, heart rhythm problems, and heart defects.
        """)
        
        # Create a pie chart for heart disease risk factors
        fig = px.pie(
            heart_data, 
            values='Importance', 
            names='Factor',
            title='Heart Disease Risk Factors',
            color_discrete_sequence=px.colors.sequential.Reds
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Parkinson's Disease")
        st.write("""
        Parkinson's disease is a brain disorder that leads to shaking, stiffness, 
        and difficulty with walking, balance, and coordination.
        """)
        
        # Create a radar chart for Parkinson's
        fig = px.line_polar(
            parkinsons_data, 
            r='Importance', 
            theta='Factor', 
            line_close=True,
            title='Parkinson\'s Disease Indicators',
            color_discrete_sequence=['purple']
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Add disclaimer
    st.markdown('<div class="info-text">', unsafe_allow_html=True)
    st.markdown("""
    **Disclaimer**: This application is for educational purposes only and should not replace professional medical advice. 
    Always consult with a healthcare provider for diagnosis and treatment.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# Diabetes prediction page
elif selected == 'Diabetes Prediction':
    st.markdown('<h1 class="main-header">Diabetes Prediction</h1>', unsafe_allow_html=True)
    
    # Brief explanation
    st.markdown('<div class="info-text">', unsafe_allow_html=True)
    st.write("""
    This tool uses machine learning to assess diabetes risk based on health parameters.
    Please fill in your information accurately for the best prediction results.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("### Enter your health information")
    
    # Create two tabs: Form and Information
    tab1, tab2 = st.tabs(["Input Form", "Parameter Information"])
    
    with tab1:
        # getting the input data from the user
        col1, col2, col3 = st.columns(3)

        with col1:
            Pregnancies = st.number_input('Number of Pregnancies', min_value=0, max_value=20, value=0)

        with col2:
            Glucose = st.number_input('Glucose Level (mg/dL)', min_value=0, max_value=300, value=100)

        with col3:
            BloodPressure = st.number_input('Blood Pressure (mm Hg)', min_value=0, max_value=200, value=70)

        with col1:
            SkinThickness = st.number_input('Skin Thickness (mm)', min_value=0, max_value=100, value=20)

        with col2:
            Insulin = st.number_input('Insulin Level (mu U/ml)', min_value=0, max_value=900, value=80)

        with col3:
            BMI = st.number_input('BMI value', min_value=0.0, max_value=70.0, value=25.0, format="%.1f")

        with col1:
            DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=3.0, value=0.5, format="%.3f")

        with col2:
            Age = st.number_input('Age', min_value=0, max_value=120, value=30)
    
    with tab2:
        st.markdown("### Understanding Diabetes Parameters")
        
        # Create an expandable section for each parameter
        with st.expander("Glucose Level"):
            st.write("""
            **What is it?** The amount of glucose (sugar) in your blood.
            
            **Normal range:** 70-99 mg/dL when fasting
            
            **Risk indicator:** Values above 126 mg/dL while fasting may indicate diabetes.
            """)
            
            # Sample visualization of glucose levels
            glucose_ranges = ['Low (<70)', 'Normal (70-99)', 'Prediabetes (100-125)', 'Diabetes (>126)']
            glucose_colors = ['#64B5F6', '#4CAF50', '#FFC107', '#F44336']
            fig = go.Figure(data=[go.Bar(
                x=glucose_ranges,
                y=[70, 30, 25, 30],
                marker_color=glucose_colors
            )])
            fig.update_layout(title_text='Blood Glucose Distribution (mg/dL)')
            st.plotly_chart(fig, use_container_width=True)
            
        with st.expander("BMI (Body Mass Index)"):
            st.write("""
            **What is it?** A measure of body fat based on height and weight.
            
            **Calculation:** Weight (kg) / [Height (m)]²
            
            **Categories:**
            - Below 18.5: Underweight
            - 18.5 - 24.9: Normal weight
            - 25 - 29.9: Overweight
            - 30 and Above: Obesity
            
            **Risk indicator:** BMI over 25 increases risk of type 2 diabetes.
            """)
            
            # BMI visualization
            bmi_ranges = ['Underweight', 'Normal', 'Overweight', 'Obese']
            bmi_values = [17, 22, 27, 33]
            fig = go.Figure()
            fig.add_trace(go.Indicator(
                mode = "gauge+number",
                value = BMI,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Your BMI"},
                gauge = {
                    'axis': {'range': [None, 40], 'tickwidth': 1},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 18.5], 'color': '#64B5F6'},
                        {'range': [18.5, 25], 'color': '#4CAF50'},
                        {'range': [25, 30], 'color': '#FFC107'},
                        {'range': [30, 40], 'color': '#F44336'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': BMI
                    }
                }
            ))
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("Other Important Parameters"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Blood Pressure**")
                st.write("Normal range: Below 120/80 mm Hg")
                st.write("Elevated levels can both affect and be affected by diabetes.")
                
                st.markdown("**Skin Thickness**")
                st.write("Measures the fat layer under the skin.")
                st.write("Related to body fat distribution and insulin resistance.")
            
            with col2:
                st.markdown("**Insulin Level**")
                st.write("Measures how much insulin is in your blood.")
                st.write("Abnormal levels may indicate insulin resistance or insufficient production.")
                
                st.markdown("**Diabetes Pedigree Function**")
                st.write("Scores genetic influence for diabetes based on family history.")
                st.write("Higher values indicate stronger genetic predisposition.")

    # Create a progress bar widget for processing time simulation
    progress_placeholder = st.empty()
    result_placeholder = st.empty()
    
    # Creating a button for Prediction
    if st.button('Run Diabetes Prediction'):
        try:
            # Prepare data for API
            input_data = {
                "Pregnancies": float(Pregnancies),
                "Glucose": float(Glucose),
                "BloodPressure": float(BloodPressure),
                "SkinThickness": float(SkinThickness),
                "Insulin": float(Insulin),
                "BMI": float(BMI),
                "DiabetesPedigreeFunction": float(DiabetesPedigreeFunction),
                "Age": float(Age)
            }
            
            # Show processing animation
            with progress_placeholder.container():
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                st.success("Analysis complete!")
            
            # Call API
            try:
                response = requests.post(f"{API_URL}/predict/diabetes", json=input_data, timeout=5)
                
                # Get result
                if response.status_code == 200:
                    result = response.json()
                    prediction = result.get("prediction", 0)
                    
                    # Show result with visualizations
                    with result_placeholder.container():
                        if prediction == 1:
                            st.markdown('<div class="result-positive">', unsafe_allow_html=True)
                            st.markdown("### Result: The Person is Diabetic")
                            st.markdown("The model predicts a higher likelihood of diabetes based on the provided parameters.")
                            st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="result-negative">', unsafe_allow_html=True)
                            st.markdown("### Result: The Person does not have Diabetes")
                            st.markdown("The model predicts a lower likelihood of diabetes based on the provided parameters.")
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Display visualization of results
                        col1, = st.columns(1)
                        
                        with col1:
                            # Parameter visualization
                            radar_fig = visualize_parameters(input_data, 'diabetes')
                            if radar_fig:
                                st.plotly_chart(radar_fig, use_container_width=True)
                        
                        # Recommendations section
                        st.markdown("### Recommendations")
                        if prediction == 1:
                            st.markdown("""
                            - 👨‍⚕️ **Consult with a healthcare provider** for proper diagnosis and management
                            - 📊 **Monitor blood glucose levels** regularly
                            - 🥗 **Maintain a healthy diet** rich in fiber, low in processed sugars
                            - 🏃‍♀️ **Exercise regularly** - aim for at least 150 minutes per week
                            - ⚖️ **Maintain a healthy weight** through proper diet and exercise
                            """)
                        else:
                            st.markdown("""
                            - 🥗 **Maintain a healthy diet** rich in vegetables, fruits, and whole grains
                            - 🏃‍♀️ **Stay physically active** with regular exercise
                            - 🩺 **Schedule regular check-ups** with your healthcare provider
                            - 💧 **Stay hydrated** and limit sugary beverages
                            - 😴 **Get adequate sleep** to help maintain healthy blood sugar levels
                            """)
                else:
                    st.error(f"Error occurred during prediction: {response.status_code}")
            except requests.exceptions.ConnectionError:
                st.error("Unable to connect to the prediction API. Please make sure the server is running.")
                
                # Provide mock results for demonstration (in case API is not available)
                with result_placeholder.container():
                    st.warning("⚠️ Using demonstration mode (API not available)")
                    
                    # Mock prediction based on glucose and BMI
                    mock_prediction = 1 if (Glucose > 125 and BMI > 30) else 0
                    mock_probability = 0.7 if mock_prediction == 1 else 0.3
                    
                    if mock_prediction == 1:
                        st.markdown('<div class="result-positive">', unsafe_allow_html=True)
                        st.markdown("### Result: Higher Risk of Diabetes (Demo)")
                        st.markdown("The model predicts a higher likelihood of diabetes based on the provided parameters.")
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="result-negative">', unsafe_allow_html=True)
                        st.markdown("### Result: Lower Risk of Diabetes (Demo)")
                        st.markdown("The model predicts a lower likelihood of diabetes based on the provided parameters.")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Display visualization of results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Gauge chart for probability
                        fig = create_gauge_chart(mock_probability, "Risk Probability (Demo)")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Parameter visualization
                        radar_fig = visualize_parameters(input_data, 'diabetes')
                        if radar_fig:
                            st.plotly_chart(radar_fig, use_container_width=True)
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Heart Disease Prediction
elif selected == 'Heart Disease Prediction':
    st.markdown('<h1 class="main-header">Heart Disease Prediction</h1>', unsafe_allow_html=True)
    
    # Brief explanation
    st.markdown('<div class="info-text">', unsafe_allow_html=True)
    st.write("""
    This tool assesses heart disease risk using multiple cardiovascular parameters.
    Enter your information accurately for the best prediction results.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Add tabs
    tab1, tab2 = st.tabs(["Input Form", "Heart Health Information"])
    
    with tab1:
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input('Age', min_value=20, max_value=100, value=45)

        with col2:
            sex = st.radio('Sex', options=[0, 1], format_func=lambda x: 'Female' if x == 0 else 'Male')

        with col3:
            cp = st.selectbox('Chest Pain Type', 
                               options=[0, 1, 2, 3], 
                               format_func=lambda x: ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'][x])

        with col1:
            trestbps = st.number_input('Resting Blood Pressure (mm Hg)', min_value=90, max_value=200, value=120)

        with col2:
            chol = st.number_input('Serum Cholesterol (mg/dl)', min_value=100, max_value=500, value=200)

        with col3:
            fbs = st.radio('Fasting Blood Sugar > 120 mg/dl', 
                            options=[0, 1], 
                            format_func=lambda x: 'No' if x == 0 else 'Yes')

        with col1:
            restecg = st.selectbox('Resting ECG Results', 
                                    options=[0, 1], 
                                    format_func=lambda x: ['Normal', 'Left Ventricular Hypertrophy'][x])

        with col2:
            thalach = st.number_input('Maximum Heart Rate', min_value=60, max_value=220, value=150)

        with col3:
            exang = st.radio('Exercise Induced Angina', 
                              options=[0, 1], 
                              format_func=lambda x: 'No' if x == 0 else 'Yes')

        with col1:
            oldpeak = st.number_input('ST Depression by Exercise', min_value=0.0, max_value=10.0, value=1.0, step=0.1)

        with col2:
            slope = st.selectbox('Slope of Peak Exercise ST Segment', 
                                  options=[0, 1, 2], 
                                  format_func=lambda x: ['Upsloping', 'Flat', 'Downsloping'][x])
            
        with col3:
            ca = st.selectbox('Number of Major Vessels Colored by Flourosopy', 
                               options=[0, 1, 2, 3, 4])

        with col1:
            thal = st.selectbox('Thalassemia', 
                                 options=[0, 1, 2, 3], 
                                 format_func=lambda x: ['Not Available', 'Normal', 'Fixed Defect', 'Reversible Defect'][x])
    
    with tab2:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Heart Disease Risk Factors")
            st.write("""
            Several factors increase your risk of heart disease:
            
            - **Age**: Risk increases with age
            - **Sex**: Men generally have higher risk
            - **High blood pressure**: Forces your heart to work harder
            - **High cholesterol**: Can lead to plaque buildup in arteries
            - **Smoking**: Damages blood vessels and reduces oxygen
            - **Diabetes**: Increases risk of heart disease
            - **Family history**: Genetic factors play a role
            - **Obesity**: Puts extra strain on your heart
            """)
            
            # Add a small visualization
            st.markdown("#### Key Heart Disease Indicators")
            fig = px.pie(
                heart_data, 
                values='Importance', 
                names='Factor',
                color_discrete_sequence=px.colors.sequential.Reds
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Understanding Heart Parameters")
            
            with st.expander("Chest Pain Types"):
                st.write("""
                **Type 0 (Typical Angina)**: Pain in your chest that's caused by reduced blood flow to your heart.
                
                **Type 1 (Atypical Angina)**: Chest pain that doesn't meet all criteria for typical angina.
                
                **Type 2 (Non-anginal Pain)**: Chest pain not related to your heart.
                
                **Type 3 (Asymptomatic)**: No chest pain symptoms.
                """)
                
            with st.expander("ST Depression & ECG Results"):
                st.write("""
                **ST Depression**: Represents the level of depression in the ST segment during an ECG.
                Higher values indicate more significant changes, often associated with ischemia.
                
                **Resting ECG Results**:
                - Normal: No abnormalities
                - Left Ventricular Hypertrophy: Thickening of heart's main pumping chamber
                """)
                
            with st.expander("Maximum Heart Rate"):
                st.write("""
                Maximum heart rate is the highest your heart rate should be during exercise.
                
                **Formula**: 220 - your age
                
                A substantially lower maximum heart rate during stress testing can indicate cardiovascular problems.
                """)
                
                # Create a gauge showing the user's max heart rate vs expected
                max_heart_rate = 220 - age
                fig = go.Figure()
                fig.add_trace(go.Indicator(
                    mode = "gauge+number+delta",
                    value = thalach,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Heart Rate"},
                    delta = {'reference': max_heart_rate, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
                    gauge = {
                        'axis': {'range': [None, max_heart_rate * 1.2]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, max_heart_rate*0.7], 'color': "green"},
                            {'range': [max_heart_rate*0.7, max_heart_rate], 'color': "yellow"},
                            {'range': [max_heart_rate, max_heart_rate*1.2], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': max_heart_rate
                        }
                    }
                ))
                fig.update_layout(height=200)
                st.plotly_chart(fig, use_container_width=True)

    # Progress bar placeholder
    progress_placeholder = st.empty()
    result_placeholder = st.empty()
    
    # Creating a button for Prediction
    if st.button('Run Heart Disease Prediction'):
        try:
            # Prepare data for API
            input_data = {
                "age": float(age),
                "sex": float(sex),
                "cp": float(cp),
                "trestbps": float(trestbps),
                "chol": float(chol),
                "fbs": float(fbs),
                "restecg": float(restecg),
                "thalach": float(thalach),
                "exang": float(exang),
                "oldpeak": float(oldpeak),
                "slope": float(slope),
                "ca": float(ca),
                "thal": float(thal)
            }
            
            # Show processing animation
            with progress_placeholder.container():
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                st.success("Analysis complete!")
            
            # Call API
            try:
                response = requests.post(f"{API_URL}/predict/heart", json=input_data, timeout=5)
                
                # Get result
                if response.status_code == 200:
                    result = response.json()
                    prediction = result.get("prediction", 0)
                    probability = result.get("probability", 0.5)  # Assuming API returns probability
                    
                    # Show result with visualizations
                    with result_placeholder.container():
                        if prediction == 1:
                            st.markdown('<div class="result-positive">', unsafe_allow_html=True)
                            st.markdown("### Result: The Person has a Heart Disease")
                            st.markdown("The model predicts a higher likelihood of heart disease based on the provided parameters.")
                            st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="result-negative">', unsafe_allow_html=True)
                            st.markdown("### Result: The Person does not have a Heart Disease")
                            st.markdown("The model predicts a lower likelihood of heart disease based on the provided parameters.")
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Display visualization of results
                        col1, = st.columns(1)
                  
                        with col1:
                            # Parameter visualization
                            radar_fig = visualize_parameters(input_data, 'heart')
                            if radar_fig:
                                st.plotly_chart(radar_fig, use_container_width=True)
                        
                        # Recommendations section
                        st.markdown("### Recommendations")
                        if prediction == 1:
                            st.markdown("""
                            - 👨‍⚕️ **Consult with a cardiologist** for proper evaluation and management
                            - 💊 **Review medications** with your healthcare provider
                            - 🥗 **Adopt a heart-healthy diet** low in sodium and saturated fats
                            - 🏃‍♀️ **Begin a supervised exercise program** appropriate for your condition
                            - 🚭 **Quit smoking** and avoid secondhand smoke
                            - 😌 **Manage stress** through relaxation techniques
                            """)
                        else:
                            st.markdown("""
                            - 🩺 **Schedule regular check-ups** to monitor heart health
                            - 🥗 **Maintain a heart-healthy diet** rich in fruits, vegetables, and whole grains
                            - 🏃‍♀️ **Exercise regularly** - aim for at least 150 minutes per week
                            - 😴 **Get adequate sleep** - 7-8 hours nightly
                            - 🧘‍♀️ **Practice stress management** through mindfulness or meditation
                            """)
                else:
                    st.error(f"Error occurred during prediction: {response.status_code}")
            except requests.exceptions.ConnectionError:
                st.error("Unable to connect to the prediction API. Please make sure the server is running.")
                
                # Provide mock results for demonstration
                with result_placeholder.container():
                    st.warning("⚠️ Using demonstration mode (API not available)")
                    
                    # Mock prediction based on age, cholesterol and chest pain
                    mock_prediction = 1 if (age > 55 and chol > 240 and cp > 1) else 0
                    mock_probability = 0.75 if mock_prediction == 1 else 0.25
                    
                    if mock_prediction == 1:
                        st.markdown('<div class="result-positive">', unsafe_allow_html=True)
                        st.markdown("### Result: Higher Risk of Heart Disease (Demo)")
                        st.markdown("The model predicts a higher likelihood of heart disease based on the provided parameters.")
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="result-negative">', unsafe_allow_html=True)
                        st.markdown("### Result: Lower Risk of Heart Disease (Demo)")
                        st.markdown("The model predicts a lower likelihood of heart disease based on the provided parameters.")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Display visualization of results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Gauge chart for probability
                        fig = create_gauge_chart(mock_probability, "Risk Probability (Demo)")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Parameter visualization
                        radar_fig = visualize_parameters(input_data, 'heart')
                        if radar_fig:
                            st.plotly_chart(radar_fig, use_container_width=True)
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Parkinsons Prediction Page
elif selected == 'Parkinsons Prediction':
    st.markdown('<h1 class="main-header">Parkinson\'s Disease Prediction</h1>', unsafe_allow_html=True)
    
    # Brief explanation
    st.markdown('<div class="info-text">', unsafe_allow_html=True)
    st.write("""
    This tool assesses Parkinson's disease risk using voice and speech pattern parameters.
    For the most accurate results, provide values from proper voice recordings analyzed with acoustic software.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Add tabs for better organization
    tab1, tab2 = st.tabs(["Input Form", "Parameter Information"])
    
    with tab1:
        # Create collapsible sections for better organization of many parameters
        with st.expander("Voice Frequency Parameters", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                fo = st.number_input('MDVP:Fo(Hz) - Average Vocal Fundamental Frequency', min_value=50.0, max_value=300.0, value=120.0)

            with col2:
                fhi = st.number_input('MDVP:Fhi(Hz) - Maximum Vocal Fundamental Frequency', min_value=50.0, max_value=500.0, value=180.0)

            with col3:
                flo = st.number_input('MDVP:Flo(Hz) - Minimum Vocal Fundamental Frequency', min_value=50.0, max_value=300.0, value=100.0)
        
        with st.expander("Jitter Parameters"):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                Jitter_percent = st.number_input('MDVP:Jitter(%) - Frequency Variation', min_value=0.0, max_value=5.0, value=0.5, format="%.6f")

            with col2:
                Jitter_Abs = st.number_input('MDVP:Jitter(Abs) - Absolute Jitter', min_value=0.0, max_value=1.0, value=0.05, format="%.6f")

            with col3:
                RAP = st.number_input('MDVP:RAP - Relative Amplitude Perturbation', min_value=0.0, max_value=1.0, value=0.03, format="%.6f")

            with col4:
                PPQ = st.number_input('MDVP:PPQ - Five-Point Period Perturbation Quotient', min_value=0.0, max_value=1.0, value=0.03, format="%.6f")
            
            with col1:
                DDP = st.number_input('Jitter:DDP - Average Perturbation', min_value=0.0, max_value=1.0, value=0.03, format="%.6f")
        
        with st.expander("Shimmer Parameters"):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                Shimmer = st.number_input('MDVP:Shimmer - Amplitude Variation', min_value=0.0, max_value=1.0, value=0.2, format="%.6f")

            with col2:
                Shimmer_dB = st.number_input('MDVP:Shimmer(dB) - Variation in dB', min_value=0.0, max_value=2.0, value=0.5, format="%.6f")

            with col3:
                APQ3 = st.number_input('Shimmer:APQ3 - Three-Point Amplitude Quotient', min_value=0.0, max_value=1.0, value=0.15, format="%.6f")
                
            with col4:
                APQ5 = st.number_input('Shimmer:APQ5 - Five-Point Amplitude Quotient', min_value=0.0, max_value=1.0, value=0.15, format="%.6f")    
            
            with col1:
                APQ = st.number_input('MDVP:APQ - Amplitude Perturbation Quotient', min_value=0.0, max_value=1.0, value=0.15, format="%.6f")

            with col2:
                DDA = st.number_input('Shimmer:DDA - Average Absolute Differences', min_value=0.0, max_value=1.0, value=0.15, format="%.6f")
        
        with st.expander("Harmonicity Parameters"):
            col1, col2 = st.columns(2)
            
            with col1:
                NHR = st.number_input('NHR - Noise to Harmonics Ratio', min_value=0.0, max_value=1.0, value=0.15, format="%.6f")

            with col2:
                HNR = st.number_input('HNR - Harmonics to Noise rRtio', min_value=0.0, max_value=40.0, value=20.0, format="%.6f")
        
        with st.expander("Nonlinear Parameters"):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                RPDE = st.number_input('RPDE - Recurrence Period Density Entropy', min_value=0.0, max_value=1.0, value=0.5, format="%.6f")

            with col2:
                DFA = st.number_input('DFA - Detrended Fluctuation Analysis', min_value=0.0, max_value=2.0, value=0.7, format="%.6f")

            with col3:
                spread1 = st.number_input('spread1 - Nonlinear Measure of Fundamental Frequency', min_value=-10.0, max_value=10.0, value=0.0, format="%.6f")

            with col4:
                spread2 = st.number_input('spread2 - Nonlinear Measure of Frequency Variation', min_value=0.0, max_value=5.0, value=2.0, format="%.6f")

            with col1:
                D2 = st.number_input('D2 - Correlation Dimension', min_value=0.0, max_value=5.0, value=2.0, format="%.6f")

            with col2:
                PPE = st.number_input('PPE - Pitch Period Entropy', min_value=0.0, max_value=1.0, value=0.5, format="%.6f")
    
    with tab2:
        st.markdown("### Understanding Parkinson's Disease Voice Parameters")
        
        st.write("""
        Parkinson's disease affects speech and voice production. The parameters used in this prediction model
        capture various aspects of voice quality, stability, and patterns that may indicate neurodegenerative changes.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Voice Frequency Measures")
            st.write("""
            **Fundamental Frequency (Fo)**: The basic frequency of voice vibration, measured in Hz.
            
            **Jitter**: Measures the cycle-to-cycle variations of fundamental frequency.
            High jitter values can indicate irregular vocal fold vibrations.
            
            **Shimmer**: Measures the cycle-to-cycle variations of waveform amplitude.
            High shimmer values may indicate voice pathology.
            """)
            
            # Add visualization of normal vs Parkinson's voice patterns
            st.markdown("##### Voice Pattern Comparison")
            
            # Sample data for visualization
            x = np.linspace(0, 2*np.pi, 100)
            y1 = np.sin(5*x) + 0.05*np.random.randn(100)  # Normal voice - more regular
            y2 = np.sin(5*x) + 0.3*np.random.randn(100)   # PD voice - more irregular
            
            fig = plt.figure(figsize=(10, 4))
            plt.plot(x, y1, 'g-', label='Normal Voice Pattern')
            plt.plot(x, y2, 'r-', label='Parkinson\'s Voice Pattern')
            plt.legend()
            plt.title('Voice Waveform Comparison')
            plt.xlabel('Time')
            plt.ylabel('Amplitude')
            st.pyplot(fig)
            
        with col2:
            st.markdown("#### Advanced Acoustic Measures")
            st.write("""
            **NHR/HNR**: Noise-to-Harmonics and Harmonics-to-Noise ratios measure the amount
            of noise in the voice. Parkinson's patients often show higher noise levels.
            
            **RPDE, DFA, D2**: These are nonlinear dynamical complexity measures that capture
            subtle changes in voice patterns that may not be apparent in traditional measures.
            
            **PPE (Pitch Period Entropy)**: Measures the impaired control of stable pitch,
            which is often affected in Parkinson's disease.
            """)
            
            # Add radar chart of typical values
            st.markdown("##### Key Parameter Differences")
            
            # Sample data showing normal vs PD values for key parameters
            categories = ['Jitter', 'Shimmer', 'NHR', 'DFA', 'PPE']
            normal_values = [0.3, 2.1, 0.11, 0.65, 0.2]
            pd_values = [0.62, 5.7, 0.28, 0.72, 0.35]
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=normal_values,
                theta=categories,
                fill='toself',
                name='Normal Voice',
                line=dict(color='green')
            ))
            
            fig.add_trace(go.Scatterpolar(
                r=pd_values,
                theta=categories,
                fill='toself',
                name='Parkinson\'s Voice',
                line=dict(color='red')
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                    )
                ),
                title="Voice Parameter Comparison",
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Progress bar placeholder
    progress_placeholder = st.empty()
    result_placeholder = st.empty()
    
    # Creating a button for Prediction
    if st.button("Run Parkinson's Disease Prediction"):
        try:
            # Prepare data for API
            input_data = {
                "fo": float(fo),
                "fhi": float(fhi),
                "flo": float(flo),
                "Jitter_percent": float(Jitter_percent),
                "Jitter_Abs": float(Jitter_Abs),
                "RAP": float(RAP),
                "PPQ": float(PPQ),
                "DDP": float(DDP),
                "Shimmer": float(Shimmer),
                "Shimmer_dB": float(Shimmer_dB),
                "APQ3": float(APQ3),
                "APQ5": float(APQ5),
                "APQ": float(APQ),
                "DDA": float(DDA),
                "NHR": float(NHR),
                "HNR": float(HNR),
                "RPDE": float(RPDE),
                "DFA": float(DFA),
                "spread1": float(spread1),
                "spread2": float(spread2),
                "D2": float(D2),
                "PPE": float(PPE)
            }
            
            # Show processing animation
            with progress_placeholder.container():
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                st.success("Analysis complete!")
            
            # Call API
            try:
                response = requests.post(f"{API_URL}/predict/parkinsons", json=input_data, timeout=5)
                
                # Get result
                if response.status_code == 200:
                    result = response.json()
                    prediction = result.get("prediction", 0)
                    probability = result.get("probability", 0.5)  # Assuming API returns probability
                    
                    # Show result with visualizations
                    with result_placeholder.container():
                        if prediction == 1:
                            st.markdown('<div class="result-positive">', unsafe_allow_html=True)
                            st.markdown("### Result:The Person has Parkinson's Disease")
                            st.markdown("The model predicts higher likelihood of Parkinson's disease based on voice parameters.")
                            st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="result-negative">', unsafe_allow_html=True)
                            st.markdown("### Result: The Person does not have Parkinson's Disease")
                            st.markdown("The model predicts lower likelihood of Parkinson's disease based on voice parameters.")
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Display visualization of results
                        col1, = st.columns(1)
                        
                        with col1:
                            # For Parkinson's, create a specialized chart showing the most critical parameters
                            key_params = {
                                'PPE': PPE, 
                                'RPDE': RPDE, 
                                'Jitter': Jitter_percent, 
                                'Shimmer': Shimmer
                            }
                            
                            # Normal thresholds (simplified for demonstration)
                            thresholds = {
                                'PPE': 0.2,
                                'RPDE': 0.4, 
                                'Jitter': 0.006, 
                                'Shimmer': 0.02
                            }
                            
                            # Create bar chart comparing values with thresholds
                            param_names = list(key_params.keys())
                            user_values = list(key_params.values())
                            threshold_values = [thresholds[param] for param in param_names]
                            
                            fig = go.Figure()
                            
                            # Add user values
                            fig.add_trace(go.Bar(
                                x=param_names,
                                y=user_values,
                                name='Your Values',
                                marker_color='blue'
                            ))
                            
                            # Add threshold line
                            fig.add_trace(go.Scatter(
                                x=param_names,
                                y=threshold_values,
                                mode='lines+markers',
                                name='Typical Threshold',
                                marker_color='red'
                            ))
                            
                            fig.update_layout(
                                title="Key Voice Parameters",
                                xaxis_title="Parameters",
                                yaxis_title="Values",
                                barmode='group',
                                height=300
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Recommendations section
                        st.markdown("### Recommendations")
                        if prediction == 1:
                            st.markdown("""
                            - 👨‍⚕️ **Consult with a neurologist** for proper evaluation
                            - 🎯 **Look into speech therapy** options - they can help with voice symptoms
                            - 🧠 **Consider physical therapy** specialized for Parkinson's
                            - 💊 **Discuss medication options** with your healthcare provider
                            - 🏊‍♀️ **Regular exercise** can help manage symptoms
                            - 🧘‍♀️ **Try yoga or tai chi** for balance and flexibility
                            """)
                        else:
                            st.markdown("""
                            - 🩺 **Continue regular health check-ups**
                            - 🏃‍♀️ **Maintain regular physical activity** for neurological health
                            - 🧠 **Keep mentally active** with puzzles and learning new skills
                            - 🥗 **Eat a balanced diet** rich in antioxidants
                            - 😴 **Ensure adequate sleep** for neurological health
                            """)
                else:
                    st.error(f"Error occurred during prediction: {response.status_code}")
            except requests.exceptions.ConnectionError:
                st.error("Unable to connect to the prediction API. Please make sure the server is running.")
                
                # Provide mock results for demonstration
                with result_placeholder.container():
                    st.warning("⚠️ Using demonstration mode (API not available)")
                    
                    # Mock prediction based on some key parameters
                    # Higher PPE, RPDE and jitter values are associated with Parkinson's
                    mock_prediction = 1 if (PPE > 0.4 and RPDE > 0.5 and Jitter_percent > 0.01) else 0
                    mock_probability = 0.8 if mock_prediction == 1 else 0.2
                    
                    if mock_prediction == 1:
                        st.markdown('<div class="result-positive">', unsafe_allow_html=True)
                        st.markdown("### Result: Higher Risk of Parkinson's Disease (Demo)")
                        st.markdown("The model predicts higher likelihood of Parkinson's disease based on voice parameters.")
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="result-negative">', unsafe_allow_html=True)
                        st.markdown("### Result: Lower Risk of Parkinson's Disease (Demo)")
                        st.markdown("The model predicts lower likelihood of Parkinson's disease based on voice parameters.")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Display visualization of results
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        # Gauge chart for probability
                        fig = create_gauge_chart(mock_probability, "Risk Probability (Demo)")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # For Parkinson's, create a specialized chart showing the most critical parameters
                        key_params = {
                            'PPE': PPE, 
                            'RPDE': RPDE, 
                            'Jitter': Jitter_percent, 
                            'Shimmer': Shimmer
                        }
                        
                        # Normal thresholds (simplified for demonstration)
                        thresholds = {
                            'PPE': 0.2,
                            'RPDE': 0.4, 
                            'Jitter': 0.006, 
                            'Shimmer': 0.02
                        }
                        
                        # Create bar chart comparing values with thresholds
                        param_names = list(key_params.keys())
                        user_values = list(key_params.values())
                        threshold_values = [thresholds[param] for param in param_names]
                        
                        fig = go.Figure()
                        
                        # Add user values
                        fig.add_trace(go.Bar(
                            x=param_names,
                            y=user_values,
                            name='Your Values',
                            marker_color='blue'
                        ))
                        
                        # Add threshold line
                        fig.add_trace(go.Scatter(
                            x=param_names,
                            y=threshold_values,
                            mode='lines+markers',
                            name='Typical Threshold',
                            marker_color='red'
                        ))
                        
                        fig.update_layout(
                            title="Key Voice Parameters",
                            xaxis_title="Parameters",
                            yaxis_title="Values",
                            barmode='group',
                            height=300
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"An error occurred: {e}")