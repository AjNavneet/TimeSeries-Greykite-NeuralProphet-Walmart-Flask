# Importing required packages
from flask import Flask
from flask_restful import Resource, Api
from . import Utils

# Create a Flask application
app = Flask(__name)
api = Api(app)

# LoanPred class for handling API requests
class LoanPred(Resource):
    def __init__(self, model):
        self.model = model

    def get(self):
        return {'ans': 'success'}

# Function to deploy the model on Flask
def init(load_path):
    # Load the machine learning model from the specified path
    uploaded_model = Utils.load_model(load_path)

    # Add the LoanPred resource to the Flask API
    api.add_resource(LoanPred, '/', resource_class_kwargs={'model': uploaded_model})

    # Run the Flask application on port 12345
    app.run(port=12345)
