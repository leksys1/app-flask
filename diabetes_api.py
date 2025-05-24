import pandas as pd 
from joblib import load
from flask import Flask, request, jsonify
from flask_cors import CORS
from category_encoders import BinaryEncoder

api = Flask(__name__)
CORS(api)

# Load model and data
model = load('diabetes_decision_tree_model.joblib')
x = pd.read_csv('diabetes.csv')

# Identify categorical features
categorical_features = [col for col in ['Pregnancies', 'Glucose', 'BloodPressure', 
                                        'SkinThickness', 'Insulin', 'BMI', 
                                        'DiabetesPedigreeFunction', 'Age'] if col in x.columns]

# Fit encoder
encoder = BinaryEncoder()
if categorical_features:
    x_encoded = encoder.fit_transform(x[categorical_features])

# Prediction route
# No encoding needed, just ensure input order matches model training

@api.post("/htf_prediction")
def predict_follow_up_check_up():
    data = request.get_json()
    input_df = pd.DataFrame([data])

    # Make sure the input has all expected columns
    expected_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                         'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    
    missing = [feat for feat in expected_features if feat not in input_df.columns]
    if missing:
        return jsonify({"error": f"Missing features: {missing}"}), 400

    # Reorder columns to expected order
    input_df = input_df[expected_features]

    prediction = model.predict_proba(input_df)
    class_labels = model.classes_

    response = []
    for prob in prediction:
        prob_dict = {str(k): round(float(v) * 100, 2) for k, v in zip(class_labels, prob)}
        response.append(prob_dict)

    return jsonify({"Prediction": response})



# Run the app
if __name__ == '__main__':
    api.run(debug=True, port=8000)
