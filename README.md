# Multiple Disease Prediction System

A web-based application for predicting multiple diseases including diabetes, heart disease, and Parkinson's disease using machine learning algorithms.

## 🚀 Live Demo

Check out the live application here: [Health Guard](https://streamlit-service-xyjm.onrender.com)  
👉 Try entering sample data to see instant health risk predictions!


## Features

- Interactive web interface built with Streamlit
- Backend API service built with FastAPI
- Machine learning models for predicting:
  - Diabetes
  - Heart Disease
  - Parkinson's Disease
- Responsive design for both desktop and mobile devices

## Architecture

This project uses a microservices architecture:
- **Frontend**: Streamlit web application for user interaction
- **Backend**: FastAPI service for handling prediction requests
- **ML Models**: Trained using Scikit-learn and saved with joblib

## 🛠 Installation and Local Setup

### Prerequisites
- Python 3.9+
- Git

### Steps to Run Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/multiple-disease-predictor.git
   cd multiple-disease-predictor
2. **Setup the API serivce**
   ```bash
   cd api_app
   pip install -r requirements.txt
   uvicorn api --reload
2. **Set up the Streamlit app**
   ```bash
   cd ../streamlit_app
   pip install -r requirements.txt
   streamlit run multiple_disease_pred.py

## 🧰 Technologies Used

- 🐍 Python: Core programming language
- 🧠 Scikit-learn: Building and training ML models
- ⚡ FastAPI: High-performance API service
- 🎈 Streamlit: Frontend UI
- 📊 Pandas & NumPy: Data manipulation
- 📉 Matplotlib & Plotly: Visualizations
- ☁️ Render: Deployment platform

## Future Improvements

- Add more disease prediction models
- Implement user authentication
- Add explanations for predictions
- Improve model accuracy with advanced algorithms
  
## 📝 License

This project is licensed under the MIT License.

### MIT License
   
