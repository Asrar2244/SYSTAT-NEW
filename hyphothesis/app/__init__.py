from flask import Flask
from app.api import api_blueprint

def create_app():
    app = Flask(__name__)

    # Register the API blueprint
    app.register_blueprint(api_blueprint, url_prefix='/hyphothesis/api')

    return app
