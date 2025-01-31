from flask import Blueprint

api_blueprint = Blueprint('api', __name__)

from .z_test_api import ztest_api

api_blueprint.register_blueprint(ztest_api)