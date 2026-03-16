# ⚽ Premier League ML & Hybrid AI Predictor

A hybrid Machine Learning and Generative AI application that predicts Premier League match outcomes. It combines a **Deep Learning Neural Network (TensorFlow)** for mathematical Expected Goals (xG) forecasting with an **Agentic AI (Gemini 2.5 Flash)** that searches the live web for context (injuries, manager changes, odds) to adjust the final predictions.

## 🎯 Architecture & Workflow
1. **Data Engineering:** Automatically fetches years of historical Premier League data (football-data.co.uk) and extracts key features like Home/Away win rates, attack power, and xG proxies.
2. **Neural Network (Regression):** A Sequential Deep Learning model trained dynamically to predict the raw Expected Goals for both the home and away teams.
3. **Agentic Post-Processing:** The base mathematical predictions are sent to a Gemini AI agent. The agent is equipped with live web-search tools to evaluate current real-world constraints (e.g., "Star striker is injured today") and output the final adjusted integer score.
4. **Interactive UI:** Built with Streamlit to cache the model training process and visualize the predictions elegantly.

## 🛠️ Tech Stack
* **Machine Learning:** TensorFlow, Keras, NumPy, Pandas
* **Generative AI:** Google Gemini API (with Google Search Tool Grounding)
* **Frontend:** Streamlit
* **APIs:** Official Fantasy Premier League (FPL) API

## 🚀 How to Run Locally

1. **Clone the repository:**
   git clone [https://github.com/ShaharB11/PremierLeague_ML_Predictor.git](https://github.com/ShaharB11/PremierLeague_ML_Predictor.git)
   cd PremierLeague_ML_Predictor
2. 
Install dependencies:
(Note: Installing TensorFlow might take a few moments)
pip install -r requirements.txt

Set up your API Key:
Open app.py and replace "YOUR_GEMINI_API_KEY_HERE" with your actual Google Gemini API key.




Launch the Streamlit App:
streamlit run app.py
Click the "Initialize System & Train Neural Network" button in the UI to start the data pipeline and model training.