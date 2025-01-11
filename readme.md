# Garments Worker Productivity Prediction

This is an assessment project for the MLDM (Machine Learning and Data Mining) course. The project involves predicting worker productivity in a garments factory based on various features.

## Built With
- **Python** (Flask for API)
- **Machine Learning** (Scikit-learn)
- **Joblib** (for saving/loading the trained model)
- **Pandas** (for data handling)

## Features
- The project uses a trained machine learning model to predict productivity based on input features like:
  - Worker-specific features (e.g., incentive, overtime, idle time)
  - Categorical features (e.g., department, team, day of the week)
  
- The API exposes a single `/predict` endpoint where you can send a POST request with the worker's data to get a productivity prediction.

## Running the API Locally
1. Clone the repository.
2. Create and activate a virtual environment:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. Run the Flask API:
    ```bash
    python api.py
    ```
5. The API will be running on `http://127.0.0.1:5000/`. You can now test it using Postman or any HTTP client.

## Deployment
This project can be deployed to platforms like Heroku or any server supporting Python Flask applications.
