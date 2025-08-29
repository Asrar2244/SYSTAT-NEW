import os

#log file paths
Z_TEST_LOG_FILE_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'logs', 'z_test_api.log')
TWO_SAMPLE_Z_TEST_LOG_FILE_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'logs', 'two_sample_z_test_api.log')
ONE_SAMPLE_T_TEST_LOG_FILE_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'logs', 'one_sample_t_test_api.log')
TWO_SAMPLE_T_TEST_LOG_FILE_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'logs', 'two_sample_t_test_api.log')
PAIRED_T_TEST_LOG_FILE_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'logs', 'paired_t_test_api.log')

#pre-defined input values
ALPHA_VALUE_DEFAULT = 0.05
YATES_CORRECTION_DEFAULT = 0
CONFIDENCE_INTERVAL_DEFAULT = 0.95
POPULATION_MEAN_DEFAULT = 0
ALTERNATIVE_DEFAULT = "two-sided"
P_VALUE_REJECT_DEFAULT = 0.05  
SAMPLE_SIZE_MIN = 2
EQUAL_VARIANCE_DEFAULT = 0.05

# Error message constants
INVALID_JSON_ERROR = "Invalid input. Please provide JSON data."
VALUE_ERROR_MSG = "Invalid input value: {}"
KEY_ERROR_MSG = "Missing required field: {}"
TYPE_ERROR_MSG = "Invalid data type: {}"
ZERO_DIVISION_ERROR_MSG = "Division by zero encountered during calculation."
INDEX_ERROR_MSG = "Error while processing data. Ensure proper data structure."
UNEXPECTED_ERROR_MSG = "An unexpected error occurred. Please try again later."

# Logging error messages
LOG_VALUE_ERROR = "ValueError: {}"
LOG_KEY_ERROR = "KeyError: {}"
LOG_TYPE_ERROR = "TypeError: {}"
LOG_ZERO_DIVISION_ERROR = "ZeroDivisionError: {}"
LOG_INDEX_ERROR = "IndexError: {}"
LOG_UNEXPECTED_ERROR = "Unexpected error: {}"

# data vars
POPULATION_MEAN = "population_mean"
CONFIDENCE_LEVEL = "confidence_level"
P_VALUE_REJECT = "p_value_reject"
ALTERNATIVE = "alternative"
ALPHA_VALUE = "alpha_value"
SHAPIRO_WILK = "shapiro_wilk"
KOLMOGOROV_SMIRNOV = "kolmogorov_smirnov"
YATES_CORRECTION = "yates_correction"
SAMPLE = "sample"
VALUES = "values"
SIZE = "size"
SAMPLE_SIZE = "sample_size"
MEAN = "mean"
DEVIATION = "deviation"
STD_ERR = "standard_error"


# result vars
T_STATISTIC = "t_statistic"
VALUES = "values"
P_VALUE = "p_value"
RESULT = "result"
CONFIDENCE_INTERVAL = "confidence_interval"
NORMALITY_TESTS = "normality_tests"
CONCLUSION_TWO_TAILED = "Conclusion_two_tailed"
CONCLUSION_ONE_TAILED = "Conclusion_one_tailed"
STATISTICAL_POWER = "statistical_power"
POWER_ONE_TAILED = "power_of_performed_two_tailed_test_with_alpha"
POWER_TWO_TAILED = "power_of_performed_one_tailed_test_with_alpha"
