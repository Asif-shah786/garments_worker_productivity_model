import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)

# Enable CORS for all domains (development only)
CORS(app)

# Load your trained model
try:
    model = joblib.load("garments_worker_productivity_model_trained.pkl")
except Exception as e:
    print("Error loading model:", e)
    model = None


# Define a function to handle one-hot encoding on the backend
def encode_features(data):
    try:
        # One-hot encode categorical features manually
        categorical_columns = ["quarter", "department", "day", "team"]

        # Assuming 'day', 'quarter', 'team' are passed as raw categorical values
        # One-hot encoding using pandas
        encoded_data = pd.DataFrame(
            columns=[
                "quarter_Quarter1",
                "quarter_Quarter2",
                "quarter_Quarter3",
                "quarter_Quarter4",
                "quarter_Quarter5",
                "department_finishing",
                "department_sweing",
                "day_Monday",
                "day_Saturday",
                "day_Sunday",
                "day_Thursday",
                "day_Tuesday",
                "day_Wednesday",
                "team_1",
                "team_2",
                "team_3",
                "team_4",
                "team_5",
                "team_6",
                "team_7",
                "team_8",
                "team_9",
                "team_10",
                "team_11",
                "team_12",
            ]
        )

        # Encoding categorical features based on the input values
        for col in categorical_columns:
            if col == "quarter":
                # Example: 'quarter' could be 1 for Quarter1, 2 for Quarter2, etc.
                quarter_data = [0] * 5
                quarter_index = int(data["quarter"]) - 1
                quarter_data[quarter_index] = 1
                encoded_data.loc[
                    0,
                    [
                        "quarter_Quarter1",
                        "quarter_Quarter2",
                        "quarter_Quarter3",
                        "quarter_Quarter4",
                        "quarter_Quarter5",
                    ],
                ] = pd.Series(quarter_data)

            elif col == "department":
                # Assuming department is either 'finishing' or 'sweing'
                if data["department"] == "finishing":
                    encoded_data.loc[0, "department_finishing"] = 1
                else:
                    encoded_data.loc[0, "department_sweing"] = 1

            elif col == "day":
                # Example: 'day' could be a string like 'Monday', 'Tuesday', etc.
                day_data = [0] * 7
                day_map = {
                    "Monday": 0,
                    "Saturday": 1,
                    "Sunday": 2,
                    "Thursday": 3,
                    "Tuesday": 4,
                    "Wednesday": 5,
                }
                day_index = day_map[data["day"]]
                day_data[day_index] = 1
                encoded_data.loc[
                    0,
                    [
                        "day_Monday",
                        "day_Saturday",
                        "day_Sunday",
                        "day_Thursday",
                        "day_Tuesday",
                        "day_Wednesday",
                    ],
                ] = pd.Series(day_data)

            elif col == "team":
                # Assuming team is a number from 1 to 12
                team_data = [0] * 12
                team_index = int(data["team"]) - 1
                team_data[team_index] = 1
                encoded_data.loc[
                    0,
                    [
                        "team_1",
                        "team_2",
                        "team_3",
                        "team_4",
                        "team_5",
                        "team_6",
                        "team_7",
                        "team_8",
                        "team_9",
                        "team_10",
                        "team_11",
                        "team_12",
                    ],
                ] = pd.Series(team_data)

        # Return the encoded data
        return encoded_data

    except Exception as e:
        raise ValueError("Error in encoding features: " + str(e))


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get the raw data from the request
        data = request.get_json()

        # Validate the input data
        required_fields = [
            "smv",
            "wip",
            "over_time",
            "incentive",
            "idle_time",
            "idle_men",
            "no_of_workers",
            "quarter",
            "department",
            "day",
            "team",
            "no_of_style_change",
        ]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400

        # Extract numerical features and encode categorical features
        numerical_features = [
            data["smv"],
            data["wip"],
            data["over_time"],
            data["incentive"],
            data["idle_time"],
            data["idle_men"],
            data["no_of_workers"],
            data["no_of_style_change"],
        ]

        # Encode categorical features (one-hot encoding)
        encoded_data = encode_features(data)

        # Combine the numerical features and encoded categorical features
        final_input = numerical_features + encoded_data.values.flatten().tolist()

        # Check if the model is loaded
        if model is None:
            return jsonify({"error": "Model not loaded correctly"}), 500

        # Make the prediction
        prediction = model.predict([final_input])

        # Convert prediction to a regular Python type to avoid serialization issue
        prediction = int(prediction[0])  # Convert to regular Python int

        # Return the prediction in a JSON format
        return jsonify({"prediction": prediction})

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
