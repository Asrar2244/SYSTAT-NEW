import os

#log file paths
Z_TEST_LOG_FILE_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'logs', 'z_test_api.log')
TWO_SAMPLE_Z_TEST_LOG_FILE_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'logs', 'two_sample_z_test_api.log')
ONE_SAMPLE_T_TEST_LOG_FILE_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'logs', 'one_sample_t_test_api.log')
TWO_SAMPLE_T_TEST_LOG_FILE_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'logs', 'two_sample_t_test_api.log')
PAIRED_T_TEST_LOG_FILE_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'logs', 'paired_t_test_api.log')

CHI_SQUARE_LOG_FILE_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'logs', 'crosstabulation', 'chi_square_api.log')
FISHER_EXACT_TEST_LOG_FILE_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'logs', 'crosstabulation', 'fisher_exact_test_api.log')
MCNEMAR_TEST_LOG_FILE_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'logs', 'crosstabulation', 'McNemars_test_api.log')
RELATIVE_RISK_LOG_FILE_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'logs', 'crosstabulation', 'Relative_risk_api.log')
ODDS_RATIO_LOG_FILE_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'logs', 'crosstabulation', 'Odds_Ratio_api.log')

SAMPLE_SIZE_TTEST_LOG_FILE_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'logs', 'Sample_size', 'Sample_size_ttest_api.log')
SAMPLE_SIZE_PAIRED_TTEST_LOG_FILE_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'logs', 'Sample_size', 'Sample_size_paired_ttest_api.log')
SAMPLE_SIZE_PROPORTION_TEST_LOG_FILE_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'logs', 'Sample_size', 'Sample_size_proportion_api.log')
SAMPLE_SIZE_ANOVA_LOG_FILE_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'logs', 'Sample_size', 'Sample_size_ANOVA_api.log')
SAMPLE_SIZE_CHISQUARE_LOG_FILE_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'logs', 'Sample_size', 'Sample_size_Chi_square_api.log')
SAMPLE_SIZE_CORRELATION_LOG_FILE_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'logs', 'Sample_size', 'Sample_size_Correlation_api.log')


TTEST_POWER_LOG_FILE_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'logs', 'Power', 'Power_t_test_api.log')
POWER_PAIRED_TTEST_LOG_FILE_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'logs', 'Power', 'Power_Paired_t_test_api.log')
PROPORTION_TEST_POWER_LOG_FILE_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'logs', 'Power', 'Power_Proportions_api.log')
ANOVA_POWER_LOG_FILE_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'logs', 'Power', 'Power_ANOVA_api.log')
CHISQUARE_POWER_LOG_FILE_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'logs', 'Power', 'Power_Chi_square_api.log')
CORRELATION_POWER_LOG_FILE_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'logs', 'Power', 'Power_Correlation_api.log')





#pre-defined input values
ALPHA_VALUE_DEFAULT = 0.05
YATES_CORRECTION_DEFAULT = 0
CONFIDENCE_INTERVAL_DEFAULT = 0.95
POPULATION_MEAN_DEFAULT = 0
ALTERNATIVE_DEFAULT = "two-sided"
P_VALUE_REJECT_DEFAULT = 0.05  
SAMPLE_SIZE_MIN = 2
EQUAL_VARIANCE_DEFAULT = 0.05

# Predefined constants for Cross Tabulation tests
EXPECTED_COUNT_THRESHOLD = 5  # Fisher’s test triggers if expected count <5
CROSS_TAB_ALPHA_DEFAULT = 0.05  # Default significance level for tests
CONTINGENCY_TABLE_MIN_SIZE = (2, 2)  # Minimum required table size for Fisher's Exact Test


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
