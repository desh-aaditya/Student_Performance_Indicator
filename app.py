from flask import Flask, request, render_template
import pandas as pd

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application


# Home page
@app.route('/')
def index():
    return render_template('index.html')


# Prediction route
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')

    try:
        # ✅ CORRECT FIELD MAPPING (VERY IMPORTANT)
        data = CustomData(
            gender=request.form["gender"],
            race_ethnicity=request.form["ethnicity"],
            parental_level_of_education=request.form["parental_level_of_education"],
            lunch=request.form["lunch"],
            test_preparation_course=request.form["test_preparation_course"],
            reading_score=float(request.form["reading_score"]),
            writing_score=float(request.form["writing_score"])
        )

        # Convert to DataFrame
        pred_df = data.get_data_as_data_frame()
        print("Input DataFrame:")
        print(pred_df)

        # Prediction
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        return render_template(
            'home.html',
            results=round(results[0], 2)
        )

    except Exception as e:
        print("Error:", e)
        return render_template(
            'home.html',
            results="Invalid input. Please fill all fields correctly."
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
