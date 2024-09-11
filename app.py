import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from llm_motor import LLMMotor
import os

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Zorg ervoor dat u uw API key veilig opslaat, bij voorkeur in een omgevingsvariabele
api_key = os.environ.get('OPENAI_API_KEY')
llm_motor = LLMMotor(api_key)

app = Flask(__name__)
CORS(app)  # Voeg CORS toe aan de app

@app.route('/api/send-message', methods=['POST'])
def send_message():
    data = request.json
    message = data.get('message')
    response = llm_motor.generate_response(message)
    return jsonify({"response": response})

@app.route('/api/start-conversation', methods=['POST'])
def start_conversation():
    opening_message = llm_motor.start_new_conversation()
    return jsonify({"message": opening_message})

@app.route('/api/available-models', methods=['GET'])
def get_available_models():
    models = llm_motor.get_available_models()
    return jsonify({"models": models})

@app.route('/api/select-model', methods=['POST'])
def select_model():
    data = request.json
    model = data.get('model')
    # Uw LLMMotor heeft geen methode om het model te selecteren, dus we sturen alleen een bevestiging terug
    # U kunt deze functionaliteit later toevoegen aan uw LLMMotor klasse indien nodig
    return jsonify({"success": True, "message": f"Model {model} geselecteerd"})

@app.route('/api/clear-memory', methods=['POST'])
def clear_memory():
    llm_motor.clear_memory()
    return jsonify({"success": True, "message": "Geheugen gewist"})

@app.route('/', methods=['GET'])
def home():
    return "Hello, Flask server is running!"

@app.route('/test', methods=['GET'])
def test():
    logger.info("Test route accessed")
    return "Test route is working!"

if __name__ == '__main__':
    logger.info("Starting Flask server on http://0.0.0.0:5001")
    app.run(debug=True, host='0.0.0.0', port=5001)