# website/routes/disease_routes.py
from flask import Blueprint, request, session, make_response
import pandas as pd

disease_routes = Blueprint('disease_routes', __name__)

@disease_routes.route('/download_csv')
def download_csv():
    user = request.args.get('user')
    disease = request.args.get('disease')
    data = session.get('last_prediction_data')

    if not data:
        return "No prediction data found in session", 400

    df = pd.DataFrame([data])
    filename = f"{user}_{disease}_report.csv"
    response = make_response(df.to_csv(index=False))
    response.headers["Content-Disposition"] = f"attachment; filename={filename}"
    response.headers["Content-Type"] = "text/csv"
    return response
