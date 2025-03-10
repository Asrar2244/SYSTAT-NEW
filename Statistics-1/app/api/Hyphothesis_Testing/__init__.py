from flask import Blueprint

api_blueprint = Blueprint('api', __name__)

from .z_test_api import ztest_api
from .two_sample_z_test_api import two_sample_ztest_api
from .one_sample_t_test_api import one_sample_t_test_api
from .two_sample_t_test_api import two_sample_t_test_api
from .paired_t_test_api import paired_t_test_api

api_blueprint.register_blueprint(ztest_api)
api_blueprint.register_blueprint(two_sample_ztest_api)
api_blueprint.register_blueprint(one_sample_t_test_api)
api_blueprint.register_blueprint(two_sample_t_test_api)
api_blueprint.register_blueprint(paired_t_test_api)