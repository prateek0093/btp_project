"""
Logging configuration for FSQ SDK.

This module provides logging utilities and configuration for the FSQ SDK.
Users can configure logging levels and handlers to suit their needs.

Usage:
    >>> from fsq_sdk.logging_config import get_logger, set_log_level
    >>> import logging
    >>> 
    >>> # Set log level for all FSQ SDK loggers
    >>> set_log_level(logging.DEBUG)
    >>> 
    >>> # Or configure specific logger
    >>> logger = get_logger('encoder')
    >>> logger.setLevel(logging.WARNING)
"""

import logging
from typing import Optional

# Base logger name for the SDK
SDK_LOGGER_NAME = 'fsq_sdk'

# Create the base SDK logger
_sdk_logger = logging.getLogger(SDK_LOGGER_NAME)

# Default to WARNING level - users can override
_sdk_logger.setLevel(logging.WARNING)

# Add a NullHandler by default to prevent "No handler found" warnings
# Users can add their own handlers
_sdk_logger.addHandler(logging.NullHandler())


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger for the FSQ SDK.
    
    Args:
        name: Optional sub-logger name. If provided, returns a child logger
              (e.g., 'encoder' returns 'fsq_sdk.encoder')
    
    Returns:
        Logger instance
    
    Example:
        >>> logger = get_logger('encoder')
        >>> logger.debug("Encoding block at position (0, 0)")
    """
    if name:
        return logging.getLogger(f"{SDK_LOGGER_NAME}.{name}")
    return _sdk_logger


def set_log_level(level: int):
    """
    Set the logging level for all FSQ SDK loggers.
    
    Args:
        level: Logging level (e.g., logging.DEBUG, logging.INFO, logging.WARNING)
    
    Example:
        >>> import logging
        >>> set_log_level(logging.DEBUG)  # Enable debug output
    """
    _sdk_logger.setLevel(level)


def enable_console_logging(level: int = logging.INFO, 
                          format_string: Optional[str] = None):
    """
    Enable console logging for the FSQ SDK with a formatted handler.
    
    This is a convenience function to quickly enable visible logging output.
    
    Args:
        level: Logging level for the console handler
        format_string: Custom format string (uses default if None)
    
    Example:
        >>> enable_console_logging(logging.DEBUG)
        >>> # Now all FSQ SDK operations will print to console
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Remove existing handlers to avoid duplicates
    for handler in _sdk_logger.handlers[:]:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.NullHandler):
            _sdk_logger.removeHandler(handler)
    
    # Create and configure console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(format_string))
    
    _sdk_logger.addHandler(console_handler)
    _sdk_logger.setLevel(level)


def disable_logging():
    """
    Disable all logging for the FSQ SDK.
    
    This sets the log level to CRITICAL+1, effectively silencing all logs.
    """
    _sdk_logger.setLevel(logging.CRITICAL + 1)
