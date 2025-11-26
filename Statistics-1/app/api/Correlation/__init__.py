
from flask import Blueprint

correlation_api_blueprint = Blueprint('correlation_api', __name__)

from .Normality_api import normality_test_api
from .pca_api import pca_api
from .pearson_api import pearson_api
from .spearman_api import spearman_api

correlation_api_blueprint.register_blueprint(normality_test_api)
correlation_api_blueprint.register_blueprint(pca_api)
correlation_api_blueprint.register_blueprint(pearson_api)
correlation_api_blueprint.register_blueprint(spearman_api)
