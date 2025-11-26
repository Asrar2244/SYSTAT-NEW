import logging
import os

class Logger:
    """
    A custom logger class to log messages at different levels (info, error, warning, debug).
    """
    def __init__(self, log_file_name, level=logging.INFO, format="%(asctime)s %(filename)s %(levelname)s %(message)s"):
        """
        Initializes the logger with a file handler to write logs to the specified file.
        
        Args:
            log_file_name (str): The name of the log file.
            level (int): The logging level (default is logging.INFO).
            format (str): The log message format (default is "%(asctime)s %(filename)s %(levelname)s %(message)s").
        """
        self.logger = logging.getLogger()  # Use the root logger
        self.logger.setLevel(level)

        formatter = logging.Formatter(format)
        file_handler = logging.FileHandler(log_file_name, mode='w')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)

    def info(self, message):
        """Log an informational message."""
        self.logger.info(message, stacklevel=2)  # Get caller’s filename

    def error(self, message):
        """Log an error message."""
        self.logger.error(message, stacklevel=2)

    def warning(self, message):
        """Log a warning message."""
        self.logger.warning(message, stacklevel=2)

    def debug(self, message):
        """Log a debug message."""
        self.logger.debug(message, stacklevel=2)

    def log_exception(self, context, message):
        """
        Log an exception with additional context.
        
        Args:
            context (str): Custom context or label for the log entry (e.g., "LOG_VALUE_ERROR").
            message (str): The error message to log.
        """
        self.logger.exception(f"{context} - {message}", stacklevel=2)

