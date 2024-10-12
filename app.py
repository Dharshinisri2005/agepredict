from flask import Flask, render_template, request, send_file
import pandas as pd
import numpy as np
import catboost  # Assuming you have CatBoost installed
import os
import pickle
import io

app = Flask(__name__)

# Load the model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if a file is uploaded
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']

    # Read the uploaded CSV file
    if file.filename.endswith('.csv'):
        data = pd.read_csv(file)

        # Extract relevant columns for prediction
        input_data = data[['P. Name', 'P. Name Kepler', 'P. Name KOI', 'P. Zone Class', 'P. Mass Class', 
                           'P. Composition Class', 'P. Atmosphere Class', 'P. Habitable Class', 
                           'P. Min Mass (EU)', 'P. Mass (EU)', 'P. Radius (EU)', 'P. Density (EU)', 
                           'P. Gravity (EU)', 'P. Esc Vel (EU)', 'P. SFlux Min (EU)', 'P. SFlux Mean (EU)', 
                           'P. SFlux Max (EU)', 'P. Teq Min (K)', 'P. Teq Mean (K)', 'P. Teq Max (K)', 
                           'P. Ts Min (K)', 'P. Ts Mean (K)', 'P. Ts Max (K)', 'P. Surf Press (EU)', 
                           'P. Mag', 'P. Appar Size (deg)', 'P. Period (days)', 'P. Sem Major Axis (AU)', 
                           'P. Eccentricity', 'P. Mean Distance (AU)', 'P. Inclination (deg)', 
                           'P. Omega (deg)', 'S. Name', 'S. Name HD', 'S. Name HIP', 
                           'S. Constellation', 'S. Type', 'S. Mass (SU)', 'S. Radius (SU)', 
                           'S. Teff (K)', 'S. Luminosity (SU)', 'S. [Fe/H]', 'S. Appar Mag', 
                           'S. Distance (pc)', 'S. RA (hrs)', 'S. DEC (deg)', 'S. Mag from Planet', 
                           'S. Size from Planet (deg)', 'S. No. Planets', 'S. No. Planets HZ', 
                           'S. Hab Zone Min (AU)', 'S. Hab Zone Max (AU)', 'P. HZD', 
                           'P. HZC', 'P. HZA', 'P. HZI', 'P. SPH', 'P. Int ESI', 
                           'P. Surf ESI', 'P. ESI', 'S. HabCat', 'P. Habitable', 
                           'P. Hab Moon', 'P. Confirmed', 'P. Disc. Method', 
                           'P. Disc. Year', 'Unnamed: 68', 'mass_to_lum_mean', 
                           'mass_lum_max', 'mass_lum_min', 'mass_lum_diff', 
                           'Stellar_gravity_constant']].values

        # Make predictions
        predicted_ages = model.predict(input_data)

        # Create a DataFrame for the predicted ages
        results_df = pd.DataFrame(predicted_ages, columns=["Predicted Star Age (billion years)"])

        # Create a CSV buffer
        output = io.StringIO()
        results_df.to_csv(output, index=False)
        output.seek(0)

        # Return the CSV as a downloadable file
        return send_file(io.BytesIO(output.getvalue().encode()), 
                         mimetype='text/csv',
                         as_attachment=True,
                         download_name='predicted_star_ages.csv')
    else:
        return "Invalid file type. Please upload a CSV file.", 400

if __name__ == '__main__':
    app.run(debug=True)
