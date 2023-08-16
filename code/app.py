from flask import Flask, request, render_template

from prediction_pipeline import PredictPipeline



app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input_date = request.form['input_date']  # Get input date from form
        forecast_days = int(request.form['forecast_days'])  # Get forecast days from form

        # Use your prediction pipeline to predict the temperature
        predict_pipeline = PredictPipeline()  # Initialize your prediction pipeline
        forecasted_temperature = predict_pipeline.predict(input_date, forecast_days)
        
        return render_template('index.html', forecasted_temperature=forecasted_temperature)

    return render_template('index.html', forecasted_temperature=None)

if __name__=='__main__':
    app.run(host='0.0.0.0',debug=True)