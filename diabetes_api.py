from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import pandas as pd
from joblib import load
from category_encoders import BinaryEncoder

# Load your trained diabetes model
model = load("diabetes_decision_tree_model.joblib")

# Load the diabetes dataset for reference / encoder fitting
x = pd.read_csv('diabetes.csv')

# Diabetes features
categorical_features = [col for col in ['Pregnancies', 'Glucose', 'BloodPressure', 
                                        'SkinThickness', 'Insulin', 'BMI', 
                                        'DiabetesPedigreeFunction', 'Age'] if col in x.columns]

# Initialize encoder (if needed)
encoder = BinaryEncoder()
if categorical_features:
    encoder.fit(x[categorical_features])

# Initialize Flask app
api = Flask(__name__)
CORS(api)  # Enable CORS globally

@api.route('/predict', methods=['POST', 'OPTIONS'])
@cross_origin(origins='*')
def predict_follow_up_check_up():
    if request.method == 'OPTIONS':
        # CORS preflight
        return '', 200

    try:
        data = request.get_json()
        input_df = pd.DataFrame([data])

        expected_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                             'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

        missing = [feat for feat in expected_features if feat not in input_df.columns]
        if missing:
            return jsonify({"error": f"Missing features: {missing}"}), 400

        # Ensure column order
        input_df = input_df[expected_features]

        # If you want to encode, uncomment:
        # input_df = encoder.transform(input_df)

        prediction = model.predict_proba(input_df)
        class_labels = model.classes_

        response = []
        for prob in prediction:
            prob_dict = {str(k): round(float(v) * 100, 2) for k, v in zip(class_labels, prob)}
            response.append(prob_dict)

        return jsonify({"Prediction": response})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    api.run(debug=True, host='0.0.0.0', port=5000)
