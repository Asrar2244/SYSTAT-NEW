from flask import Flask
from .api.Hyphothesis_Testing import api_blueprint
from .api.CrossTabulation import crossTabulation_api_blueprint
from .api.Sample_size import sample_size_api_blueprint
from .api.Power import Power_api_blueprint


def create_app():
    app = Flask(__name__)

    app.register_blueprint(api_blueprint, url_prefix='/hyphothesis/api')
    app.register_blueprint(crossTabulation_api_blueprint, url_prefix='/cross_tabulation/api')
    app.register_blueprint(sample_size_api_blueprint, url_prefix='/sample_size/api')
    app.register_blueprint(Power_api_blueprint, url_prefix='/power/api')

    return app 