# -- coding: utf-8 --
"""Cancer Cell Classifier - FastAPI Web Interface"""

import numpy as np
import pickle
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse

# Load the trained model
with open('cancer_classifier_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

# Initialize FastAPI app
app = FastAPI()

# HTML form for input values
html_form = """
<!DOCTYPE html>
<html>
<head>
    <title>Cancer Cell Classifier</title>
</head>
<body>
    <h2>Enter Cell Features to Classify</h2>
    <form action="/classify" method="post">
        Clump Thickness (1-10): <input type="number" name="clump" min="1" max="10" required><br>
        Uniformity of Cell Size (1-10): <input type="number" name="unif_size" min="1" max="10" required><br>
        Uniformity of Cell Shape (1-10): <input type="number" name="unif_shape" min="1" max="10" required><br>
        Marginal Adhesion (1-10): <input type="number" name="marg_adh" min="1" max="10" required><br>
        Single Epithelial Cell Size (1-10): <input type="number" name="sing_epi_size" min="1" max="10" required><br>
        Bare Nuclei (1-10): <input type="number" name="bare_nuc" min="1" max="10" required><br>
        Bland Chromatin (1-10): <input type="number" name="bland_chrom" min="1" max="10" required><br>
        Normal Nucleoli (1-10): <input type="number" name="norm_nucl" min="1" max="10" required><br>
        Mitoses (1-10): <input type="number" name="mit" min="1" max="10" required><br><br>
        <input type="submit" value="Classify">
    </form>
</body>
</html>
"""

# Route to display the form
@app.get('/', response_class=HTMLResponse)
async def get_form():
    return html_form

# Route to handle form submission and make prediction
@app.post('/classify', response_class=HTMLResponse)
async def classify_cell(
    clump: int = Form(...),
    unif_size: int = Form(...),
    unif_shape: int = Form(...),
    marg_adh: int = Form(...),
    sing_epi_size: int = Form(...),
    bare_nuc: int = Form(...),
    bland_chrom: int = Form(...),
    norm_nucl: int = Form(...),
    mit: int = Form(...)):

    # Prepare the input data
    input_data = np.array([[clump, unif_size, unif_shape, marg_adh, sing_epi_size, bare_nuc, bland_chrom, norm_nucl, mit]])

    # Make prediction
    prediction = loaded_model.predict(input_data)

    # Display the result
    result = "Benign (2)" if prediction[0] == 2 else "Malignant (4)"

    response_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Classification Result</title>
    </head>
    <body>
        <h2>Classification Result</h2>
        <p>The cell is classified as: <strong>{result}</strong></p>
        <a href="/">Classify Another Cell</a>
    </body>
    </html>
    """
    return response_html

# To run the FastAPI app, save this file and run the following command in your terminal:
# uvicorn filename:app --reload