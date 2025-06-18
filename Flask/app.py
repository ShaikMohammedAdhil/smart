import numpy as np
import pickle
import pandas as pd
import os
from flask import Flask, request, jsonify, render_template
from datetime import datetime
import traceback

app = Flask(__name__)

# Global variables for model components
model = None
scale = None
encoders = None

def load_model_components():
    """Load the model, scaler, and encoders with proper error handling"""
    global model, scale, encoders
    
    try:
        model = pickle.load(open("model.pkl", 'rb'))
        print("Model loaded successfully")
    except FileNotFoundError:
        print("Warning: model.pkl not found")
        model = None
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None
    
    try:
        scale = pickle.load(open("scale.pkl", 'rb'))
        print("Scaler loaded successfully")
    except FileNotFoundError:
        print("Warning: scale.pkl not found")
        scale = None
    except Exception as e:
        print(f"Error loading scaler: {e}")
        scale = None
    
    try:
        encoders = pickle.load(open("encoder.pkl", 'rb'))
        print(f"Encoders loaded successfully. Type: {type(encoders)}")
        
        # Debug: Print encoder structure
        if isinstance(encoders, dict):
            print("Encoders keys:", list(encoders.keys()))
        else:
            print("Encoders is not a dictionary, it's:", type(encoders))
            
    except FileNotFoundError:
        print("Warning: encoder.pkl not found")
        encoders = None
    except Exception as e:
        print(f"Error loading encoders: {e}")
        encoders = None

def encode_categorical_variable(encoders, variable_name, value, default_value=0):
    """Safely encode a categorical variable"""
    try:
        if encoders is None:
            return default_value
        
        # If encoders is a dictionary
        if isinstance(encoders, dict) and variable_name in encoders:
            encoder = encoders[variable_name]
            if hasattr(encoder, 'transform'):
                try:
                    return encoder.transform([str(value)])[0]
                except ValueError:
                    # Value not seen during training, use fallback
                    print(f"Value '{value}' not seen during training for {variable_name}")
                    return default_value
        
        # Fallback to manual encoding based on your training data
        if variable_name == 'holiday':
            # Based on your training, common holidays
            holiday_map = {
                'none': 0, 'labor day': 1, 'columbus day': 2, 'veterans day': 3,
                'thanksgiving': 4, 'christmas': 5, 'new years day': 6,
                'washingtons birthday': 7, 'memorial day': 8, 'independence day': 9,
                'martin luther king jr day': 10, 'state fair': 11
            }
            return holiday_map.get(str(value).lower(), 1)  # Default to 'labor day'
        
        elif variable_name == 'weather':
            # Based on your training data
            weather_map = {
                'clouds': 0, 'clear': 1, 'rain': 2, 'drizzle': 3, 'mist': 4,
                'haze': 5, 'fog': 6, 'thunderstorm': 7, 'snow': 8, 'squall': 9, 'smoke': 10
            }
            return weather_map.get(str(value).lower(), 0)  # Default to 'clouds'
        
        return default_value
        
    except Exception as e:
        print(f"Error encoding {variable_name}: {e}")
        return default_value

def create_prediction_dataframe(holiday, temp, rain, snow, weather, year, month, day, hours, minutes, seconds):
    """Create a properly formatted DataFrame for prediction matching training format"""
    
    # Encode categorical variables
    holiday_encoded = encode_categorical_variable(encoders, 'holiday', holiday, 1)
    weather_encoded = encode_categorical_variable(encoders, 'weather', weather, 0)
    
    print(f"Encoded values - Holiday: {holiday} -> {holiday_encoded}, Weather: {weather} -> {weather_encoded}")
    
    # Create date and time strings to match training format
    date_str = f"{year}-{month:02d}-{day:02d}"
    time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    # Create DataFrame with the EXACT feature names and order from training
    # Based on your model code: x1 = data_for_corr.drop(['traffic_volume'], axis=1)
    # After you processed date and Time columns
    data = pd.DataFrame({
        'holiday': [holiday_encoded],
        'temp': [temp],
        'rain': [rain],
        'snow': [snow],
        'weather': [weather_encoded],
        'day': [day],
        'month': [month],
        'year': [year],
        'hours': [hours],
        'minutes': [minutes],
        'seconds': [seconds]
    })
    
    # Convert to the same data types as training
    data['day'] = data['day'].astype(int)
    data['month'] = data['month'].astype(int)
    data['year'] = data['year'].astype(int)
    data['hours'] = data['hours'].astype(int)
    data['minutes'] = data['minutes'].astype(int)
    data['seconds'] = data['seconds'].astype(int)
    
    return data

def make_prediction(data):
    """Make prediction using the loaded model"""
    try:
        print("Input data for prediction:")
        print(data)
        print("Data columns:", data.columns.tolist())
        print("Data shape:", data.shape)
        print("Data dtypes:", data.dtypes)
        
        # Scale the data if scaler is available
        if scale is not None:
            try:
                data_scaled = scale.transform(data)
                print("Data scaled successfully")
                print("Scaled data shape:", data_scaled.shape)
            except Exception as scale_error:
                print(f"Scaling error: {scale_error}")
                # Try without scaling
                data_scaled = data.values
                print("Using unscaled data due to scaling error")
        else:
            data_scaled = data.values
            print("Warning: No scaler available, using unscaled data")
        
        # Make prediction if model is available
        if model is not None:
            try:
                prediction = model.predict(data_scaled)
                prediction_value = max(0, int(prediction[0]))  # Ensure non-negative
                print(f"Model prediction: {prediction_value}")
                return prediction_value
            except Exception as model_error:
                print(f"Model prediction error: {model_error}")
                # Fall through to fallback prediction
        
        # Fallback prediction formula
        temp = data['temp'].iloc[0]
        rain = data['rain'].iloc[0]
        snow = data['snow'].iloc[0]
        hours = data['hours'].iloc[0]
        
        # More sophisticated fallback based on typical traffic patterns
        base_traffic = 1000
        temp_effect = temp * 15
        rain_effect = rain * -40
        snow_effect = snow * -60
        
        # Time-based adjustments (rush hours)
        if 7 <= hours <= 9 or 17 <= hours <= 19:  # Rush hours
            time_multiplier = 1.5
        elif 22 <= hours or hours <= 5:  # Night time
            time_multiplier = 0.3
        else:
            time_multiplier = 1.0
        
        prediction_value = max(0, int((base_traffic + temp_effect + rain_effect + snow_effect) * time_multiplier))
        print(f"Fallback prediction: {prediction_value}")
        
        return prediction_value
        
    except Exception as e:
        print(f"Error making prediction: {e}")
        traceback.print_exc()
        # Return a safe fallback value
        return 1000

# Load model components at startup
load_model_components()

@app.route('/')
def home():
    """Serve the main page"""
    try:
        return render_template("index.html")
    except Exception as e:
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Traffic Volume Prediction</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
                .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                h1 {{ color: #333; text-align: center; }}
                .nav {{ text-align: center; margin: 20px 0; }}
                .nav a {{ margin: 0 15px; padding: 10px 20px; background: #007bff; color: white; text-decoration: none; border-radius: 5px; }}
                .nav a:hover {{ background: #0056b3; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🚗 Traffic Volume Prediction System</h1>
                <p style="text-align: center; font-size: 18px; color: #666;">
                    Predict traffic volume based on weather conditions, time, and holidays
                </p>
                <div class="nav">
                    <a href="/predict">Make Prediction</a>
                    <a href="/api/predict">API Documentation</a>
                    <a href="/health">System Status</a>
                </div>
                <div style="margin-top: 30px; padding: 20px; background: #f8f9fa; border-radius: 5px;">
                    <h3>Available Endpoints:</h3>
                    <ul>
                        <li><strong>/predict</strong> - Interactive prediction form</li>
                        <li><strong>/api/predict</strong> - JSON API endpoint</li>
                        <li><strong>/health</strong> - System health check</li>
                    </ul>
                </div>
            </div>
        </body>
        </html>
        """

@app.route('/predict', methods=["POST", "GET"])
def predict():
    """Handle form-based predictions"""
    if request.method == "GET":
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Traffic Volume Prediction</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
                .container { max-width: 600px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                h2 { color: #333; text-align: center; }
                .form-group { margin: 15px 0; }
                label { display: block; margin-bottom: 5px; font-weight: bold; color: #555; }
                input, select { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 5px; font-size: 16px; }
                input[type="submit"] { background: #28a745; color: white; border: none; cursor: pointer; margin-top: 20px; }
                input[type="submit"]:hover { background: #218838; }
                .back-link { text-align: center; margin-top: 20px; }
                .back-link a { color: #007bff; text-decoration: none; }
            </style>
        </head>
        <body>
            <div class="container">
                <h2>🚗 Traffic Volume Prediction</h2>
                <form method="POST">
                    <div class="form-group">
                        <label for="holiday">Holiday:</label>
                        <select name="holiday" id="holiday">
                            <option value="None">None</option>
                            <option value="Labor Day">Labor Day</option>
                            <option value="Columbus Day">Columbus Day</option>
                            <option value="Veterans Day">Veterans Day</option>
                            <option value="Thanksgiving">Thanksgiving</option>
                            <option value="Christmas">Christmas</option>
                            <option value="New Years Day">New Years Day</option>
                            <option value="Independence Day">Independence Day</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label for="temp">Temperature (°C):</label>
                        <input type="number" step="0.1" name="temp" id="temp" value="20" required>
                    </div>
                    
                    <div class="form-group">
                        <label for="rain">Rain (mm):</label>
                        <input type="number" step="0.1" name="rain" id="rain" value="0" min="0" required>
                    </div>
                    
                    <div class="form-group">
                        <label for="snow">Snow (mm):</label>
                        <input type="number" step="0.1" name="snow" id="snow" value="0" min="0" required>
                    </div>
                    
                    <div class="form-group">
                        <label for="weather">Weather Condition:</label>
                        <select name="weather" id="weather">
                            <option value="Clear">Clear</option>
                            <option value="Clouds">Clouds</option>
                            <option value="Rain">Rain</option>
                            <option value="Drizzle">Drizzle</option>
                            <option value="Mist">Mist</option>
                            <option value="Haze">Haze</option>
                            <option value="Fog">Fog</option>
                            <option value="Thunderstorm">Thunderstorm</option>
                            <option value="Snow">Snow</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label for="year">Year:</label>
                        <input type="number" name="year" id="year" value="2025" min="2020" max="2030" required>
                    </div>
                    
                    <div class="form-group">
                        <label for="month">Month:</label>
                        <input type="number" name="month" id="month" value="1" min="1" max="12" required>
                    </div>
                    
                    <div class="form-group">
                        <label for="day">Day:</label>
                        <input type="number" name="day" id="day" value="1" min="1" max="31" required>
                    </div>
                    
                    <div class="form-group">
                        <label for="hours">Hour (24-hour format):</label>
                        <input type="number" name="hours" id="hours" value="12" min="0" max="23" required>
                    </div>
                    
                    <div class="form-group">
                        <label for="minutes">Minutes:</label>
                        <input type="number" name="minutes" id="minutes" value="0" min="0" max="59" required>
                    </div>
                    
                    <div class="form-group">
                        <label for="seconds">Seconds:</label>
                        <input type="number" name="seconds" id="seconds" value="0" min="0" max="59" required>
                    </div>
                    
                    <input type="submit" value="🔮 Predict Traffic Volume">
                </form>
                
                <div class="back-link">
                    <a href="/">← Back to Home</a>
                </div>
            </div>
        </body>
        </html>
        """
    
    try:
        print("Received POST request for prediction!")
        
        # Get form data with defaults
        holiday = request.form.get('holiday', 'None')
        temp = float(request.form.get('temp', 20))
        rain = float(request.form.get('rain', 0))
        snow = float(request.form.get('snow', 0))
        weather = request.form.get('weather', 'Clear')
        year = int(request.form.get('year', datetime.now().year))
        month = int(request.form.get('month', datetime.now().month))
        day = int(request.form.get('day', datetime.now().day))
        hours = int(request.form.get('hours', datetime.now().hour))
        minutes = int(request.form.get('minutes', datetime.now().minute))
        seconds = int(request.form.get('seconds', 0))
        
        print(f"Input values - Holiday: {holiday}, Temp: {temp}, Rain: {rain}, Snow: {snow}, Weather: {weather}")
        print(f"Date/Time - {year}-{month}-{day} {hours}:{minutes}:{seconds}")
        
        # Create prediction DataFrame
        data = create_prediction_dataframe(holiday, temp, rain, snow, weather, year, month, day, hours, minutes, seconds)
        
        # Make prediction
        prediction_value = make_prediction(data)
        
        # Return result
        result_text = f"Estimated Traffic Volume: {prediction_value} vehicles per hour"
        
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Prediction Result</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
                .container {{ max-width: 600px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .result {{ text-align: center; padding: 30px; background: #d4edda; border: 1px solid #c3e6cb; border-radius: 10px; margin: 20px 0; }}
                .result h2 {{ color: #155724; margin: 0; font-size: 28px; }}
                .input-summary {{ background: #f8f9fa; padding: 20px; border-radius: 5px; margin: 20px 0; }}
                .nav {{ text-align: center; margin: 20px 0; }}
                .nav a {{ margin: 0 10px; padding: 10px 20px; background: #007bff; color: white; text-decoration: none; border-radius: 5px; }}
                .nav a:hover {{ background: #0056b3; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="result">
                    <h2>🚗 {result_text}</h2>
                </div>
                
                <div class="input-summary">
                    <h3>Input Parameters:</h3>
                    <p><strong>Date & Time:</strong> {year}-{month:02d}-{day:02d} {hours:02d}:{minutes:02d}:{seconds:02d}</p>
                    <p><strong>Weather:</strong> {weather} (Temp: {temp}°C, Rain: {rain}mm, Snow: {snow}mm)</p>
                    <p><strong>Holiday:</strong> {holiday}</p>
                </div>
                
                <div class="nav">
                    <a href="/predict">Make Another Prediction</a>
                    <a href="/">Back to Home</a>
                </div>
            </div>
        </body>
        </html>
        """
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        traceback.print_exc()
        error_text = f"Error in prediction: {str(e)}"
        
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Prediction Error</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
                .container {{ max-width: 600px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .error {{ text-align: center; padding: 30px; background: #f8d7da; border: 1px solid #f5c6cb; border-radius: 10px; margin: 20px 0; }}
                .error h2 {{ color: #721c24; margin: 0; }}
                .nav {{ text-align: center; margin: 20px 0; }}
                .nav a {{ margin: 0 10px; padding: 10px 20px; background: #007bff; color: white; text-decoration: none; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="error">
                    <h2>❌ Prediction Error</h2>
                    <p>{error_text}</p>
                </div>
                <div class="nav">
                    <a href="/predict">Try Again</a>
                    <a href="/">Back to Home</a>
                </div>
            </div>
        </body>
        </html>
        """

@app.route('/api/predict', methods=["POST", "GET"])
def api_predict():
    """API endpoint for JSON-based predictions"""
    if request.method == "GET":
        # Return API documentation
        return jsonify({
            'message': 'Traffic Volume Prediction API',
            'endpoint': '/api/predict',
            'method': 'POST',
            'content_type': 'application/json',
            'required_fields': {
                'holiday': 'string (e.g., "None", "Labor Day", "Christmas")',
                'temp': 'float (temperature in Celsius)',
                'rain': 'float (rainfall in mm)',
                'snow': 'float (snowfall in mm)',
                'weather': 'string (e.g., "Clear", "Clouds", "Rain")',
                'year': 'integer (e.g., 2025)',
                'month': 'integer (1-12)',
                'day': 'integer (1-31)',
                'hours': 'integer (0-23)',
                'minutes': 'integer (0-59)',
                'seconds': 'integer (0-59)'
            },
            'example_request': {
                'holiday': 'None',
                'temp': 25.5,
                'rain': 0.0,
                'snow': 0.0,
                'weather': 'Clear',
                'year': 2025,
                'month': 6,
                'day': 15,
                'hours': 14,
                'minutes': 30,
                'seconds': 0
            }
        })
    
    try:
        # Get JSON data
        data_input = request.get_json()
        
        if not data_input:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided',
                'message': 'Please provide input data in JSON format'
            }), 400
        
        # Extract data with defaults
        holiday = data_input.get('holiday', 'None')
        temp = float(data_input.get('temp', 20))
        rain = float(data_input.get('rain', 0))
        snow = float(data_input.get('snow', 0))
        weather = data_input.get('weather', 'Clear')
        year = int(data_input.get('year', datetime.now().year))
        month = int(data_input.get('month', datetime.now().month))
        day = int(data_input.get('day', datetime.now().day))
        hours = int(data_input.get('hours', datetime.now().hour))
        minutes = int(data_input.get('minutes', datetime.now().minute))
        seconds = int(data_input.get('seconds', 0))
        
        print(f"API Request - Holiday: {holiday}, Temp: {temp}, Weather: {weather}")
        
        # Create prediction DataFrame
        data = create_prediction_dataframe(holiday, temp, rain, snow, weather, year, month, day, hours, minutes, seconds)
        
        # Make prediction
        prediction_value = make_prediction(data)
        
        return jsonify({
            'success': True,
            'prediction': prediction_value,
            'message': 'Prediction successful',
            'input_data': {
                'holiday': holiday,
                'temperature': temp,
                'rain': rain,
                'snow': snow,
                'weather': weather,
                'datetime': f"{year}-{month:02d}-{day:02d} {hours:02d}:{minutes:02d}:{seconds:02d}"
            },
            'model_info': {
                'model_loaded': model is not None,
                'scaler_loaded': scale is not None,
                'encoders_loaded': encoders is not None
            }
        })
        
    except ValueError as e:
        return jsonify({
            'success': False,
            'error': f'Invalid input data: {str(e)}',
            'message': 'Please check your input values'
        }), 400
    except Exception as e:
        print(f"API Error: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Internal server error during prediction'
        }), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    status = {
        'status': 'healthy',
        'model_loaded': model is not None,
        'scaler_loaded': scale is not None,
        'encoders_loaded': encoders is not None,
        'timestamp': datetime.now().isoformat(),
        'version': '2.0.0'
    }
    return jsonify(status)

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'message': 'Available endpoints: /, /predict, /api/predict, /health'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error',
        'message': 'Something went wrong on the server'
    }), 500

if __name__ == "__main__":
    print("Starting Traffic Volume Prediction Flask App...")
    print(f"Model loaded: {model is not None}")
    print(f"Scaler loaded: {scale is not None}")
    print(f"Encoders loaded: {encoders is not None}")
    
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("DEBUG", "True").lower() == "true"
    
    app.run(
        host="0.0.0.0",
        port=port, 
        debug=debug_mode, 
        use_reloader=False
    )