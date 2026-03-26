# 🏥 Health Guard – AI-Powered Disease Prediction System

A full-stack ML web application that predicts **Diabetes**, **Heart Disease**, and **Parkinson's Disease** — now with an AI Health Chatbot, BMI Calculator, Risk Score Explanations, and Health History tracking.

## 🚀 Live Demo

👉 **[health-guard on Hugging Face Spaces](https://huggingface.co/spaces/gourikam/health-guard)**

Try entering sample data to get instant health risk predictions with AI-generated explanations!

---

## ✨ Features

### 🔬 Disease Prediction
- Machine learning models for **Diabetes**, **Heart Disease**, and **Parkinson's Disease**
- Real-time inference via a FastAPI backend
- Interactive visualisations (radar charts, bar charts) of input parameters
- Personalised health recommendations based on the prediction result

### 🔍 Risk Score Explanation
- After every prediction, **Groq's LLaMA 3.3-70b** automatically explains *why* the model predicted what it did — in plain English, based on your exact input values

### 🤖 AI Health Chatbot
- Powered by **Groq API (LLaMA 3.3-70b)**
- Answers questions about symptoms, risk factors, prevention, and lifestyle tips
- Maintains full conversation history within a session
- Quick-question buttons for instant answers

### ⚖️ BMI Calculator
- Metric and Imperial support
- Gauge chart with colour-coded BMI categories
- Healthy weight range for your exact height
- Groq-generated personalised tips

### 📋 Health History
- Logs every prediction made during your session with a timestamp
- Shows input values used for each prediction
- Summary count per disease type
- Clear history button

---

## 🏗 Architecture

```
User → Hugging Face Spaces (Streamlit frontend)
              ↓
        Render (FastAPI backend) → ML Models (Scikit-learn)
              ↓
         Groq API (LLaMA 3.3-70b) → Chatbot + Explanations
```

This project uses a **microservices architecture**:
- **Frontend**: Streamlit app on Hugging Face Spaces
- **Backend**: FastAPI service on Render
- **AI Layer**: Groq API for chatbot and risk explanations
- **ML Models**: Trained with Scikit-learn, serialised with joblib

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit, Plotly, Matplotlib |
| Backend | FastAPI, Uvicorn |
| ML | Scikit-learn, Pandas, NumPy, joblib |
| AI/LLM | Groq API (LLaMA 3.3-70b) |
| Deployment | Hugging Face Spaces, Render |
| Version Control | Git, GitHub |

---

## 🖥 Local Setup

### Prerequisites
- Python 3.9+
- Git
- Groq API key (free at [console.groq.com](https://console.groq.com))

### Steps

**1. Clone the repository**
```bash
git clone https://github.com/gourikam/multiple-disease-predictor.git
cd multiple-disease-predictor
```

**2. Start the FastAPI backend** (Terminal 1)
```bash
cd api_app
pip install -r requirements.txt
uvicorn api:app --reload --port 10000
```

**3. Set up Groq API key**

Create `streamlit_app/.streamlit/secrets.toml`:
```toml
GROQ_API_KEY = "gsk_your_key_here"
```

**4. Start the Streamlit frontend** (Terminal 2)
```bash
cd streamlit_app
pip install -r requirements.txt
streamlit run multiple_disease_pred.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📁 Project Structure

```
multiple-disease-predictor/
├── api_app/
│   ├── api.py                  # FastAPI backend
│   ├── saved_models/           # Trained .sav model files
│   └── requirements.txt
├── streamlit_app/
│   ├── multiple_disease_pred.py  # Main Streamlit app
│   └── requirements.txt
├── render.yaml                 # Render deployment config
└── README.md
```

---

## 🔮 Future Improvements

- [ ] Add more disease prediction models (e.g. kidney disease, liver disease)
- [ ] User authentication and persistent health history
- [ ] Export prediction report as PDF
- [ ] Improve model accuracy with ensemble methods
- [ ] Voice input for accessibility

---

## ⚠️ Disclaimer

This application is for **educational purposes only** and should not replace professional medical advice. Always consult a qualified healthcare provider for diagnosis and treatment.

---

## 📝 License

This project is licensed under the [MIT License](LICENSE).

---

*Built by [Gourika Makhija](https://github.com/gourikam)*
   
