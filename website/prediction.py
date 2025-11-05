from flask import Blueprint, render_template, request, session, current_app
from .app_functions import ValuePredictor
import pandas as pd

prediction = Blueprint('prediction', __name__)

@prediction.route('/')
def index():
    return render_template('index.html')

@prediction.route('/predict', methods=['POST'])
def predict():
    try:
        # Define input fields for each disease
        disease_inputs = {
            'heart': ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                      'restecg', 'thalach', 'exang', 'oldpeak', 'slope',
                      'ca', 'thal'],
            'liver': ['age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin',
                      'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
                      'Aspartate_Aminotransferase', 'Total_Protiens',
                      'Albumin', 'Albumin_and_Globulin_Ratio'],
            'kidney': ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba',
                       'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc',
                       'rc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane'],
            'stroke': ['Gender', 'age', 'hypertension', 'heart_disease',
                       'ever_married', 'work_type', 'Residence_type',
                       'avg_glucose_level', 'bmi', 'smoking_status'],
            'diabetes': ['pregnancies', 'Glucose', 'blood_pressure', 'BSkinThickness',
                         'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        }

        # ✅ Step 1: Get disease type from form
        matched_disease = request.form.get("disease")
        if not matched_disease or matched_disease not in disease_inputs:
            raise ValueError("Unknown or missing disease type in form submission.")

        expected_order = disease_inputs[matched_disease]

        # ✅ Step 2: Collect input data in correct order
        input_data = {key: request.form[key] for key in expected_order}
        to_predict_list = [float(input_data[key]) for key in expected_order]

        # ✅ Step 3: Get prediction
        prediction_value, page = ValuePredictor(matched_disease, to_predict_list)
        session['last_prediction_data'] = input_data

        # ✅ Step 4: Prepare UI message
        css_class = "alert-danger" if prediction_value == 1 else "alert-success"
        message = "⚠️ High risk detected" if prediction_value == 1 else "✅ No significant risk detected"

        # ✅ Step 5: Save prediction to MongoDB
        logged_user = session.get('user')
        if logged_user:
            predictions_collection = current_app.mongo_db.predictions
            predictions_collection.insert_one({
                'user': logged_user,
                'disease': matched_disease,
                'prediction_value': int(prediction_value),
                'input_data': input_data,
                'timestamp': pd.Timestamp.now()
            })

        # ✅ Step 6: Render disease-specific report page
        return render_template(f"{page}_report.html",
                               input_data=input_data,
                               prediction_text=message,
                               prediction_class=css_class,
                               prediction_value=prediction_value)

    except ValueError as ve:
        return f"Invalid input: {ve}", 400
    except FileNotFoundError as fe:
        return f"Model file not found: {fe}", 500
    except Exception as e:
        return f"An unexpected error occurred: {e}", 500
