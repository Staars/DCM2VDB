"""Utility functions and classes for DICOM import"""

from typing import Optional
import os
import datetime

# Log levels
LOG_LEVEL_DEBUG = 0
LOG_LEVEL_INFO = 1
LOG_LEVEL_WARNING = 2
LOG_LEVEL_ERROR = 3

# Simple logging wrapper since Blender's logging API isn't fully implemented yet
class SimpleLogger:
    """
    Simple logger with configurable log levels and optional file output.
    
    Usage:
        log = SimpleLogger(level='INFO')
        log.debug("This won't print")
        log.info("This will print")
        
        # With file logging
        log = SimpleLogger(level='DEBUG', log_file='/tmp/dicom.log')
    """
    
    def __init__(self, level: str = 'INFO', log_file: Optional[str] = None):
        """
        Initialize logger.
        
        Args:
            level: Minimum log level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
            log_file: Optional path to log file for persistent logging
        """
        self.level_map = {
            'DEBUG': LOG_LEVEL_DEBUG,
            'INFO': LOG_LEVEL_INFO,
            'WARNING': LOG_LEVEL_WARNING,
            'ERROR': LOG_LEVEL_ERROR
        }
        self.level = self.level_map.get(level.upper(), LOG_LEVEL_INFO)
        self.log_file = log_file
        
        # Create log file if specified
        if self.log_file:
            try:
                # Ensure directory exists
                log_dir = os.path.dirname(self.log_file)
                if log_dir and not os.path.exists(log_dir):
                    os.makedirs(log_dir)
                
                # Write header
                with open(self.log_file, 'a') as f:
                    f.write(f"\n{'='*60}\n")
                    f.write(f"Log started: {datetime.datetime.now()}\n")
                    f.write(f"{'='*60}\n")
            except Exception as e:
                print(f"WARNING: Could not create log file {self.log_file}: {e}")
                self.log_file = None
    
    def _log(self, level: int, level_name: str, msg: str) -> None:
        """Internal logging method."""
        if level >= self.level:
            timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
            formatted_msg = f"{timestamp} {level_name:8} | {msg}"
            print(formatted_msg)
            
            # Write to file if enabled
            if self.log_file:
                try:
                    with open(self.log_file, 'a') as f:
                        f.write(formatted_msg + '\n')
                except Exception:
                    pass  # Silently fail file writes to avoid disrupting execution
    
    def debug(self, msg: str) -> None:
        """Log debug message (lowest priority)."""
        self._log(LOG_LEVEL_DEBUG, "DEBUG", msg)
    
    def info(self, msg: str) -> None:
        """Log info message (normal priority)."""
        self._log(LOG_LEVEL_INFO, "INFO", msg)
    
    def warning(self, msg: str) -> None:
        """Log warning message (high priority)."""
        self._log(LOG_LEVEL_WARNING, "WARNING", msg)
    
    def error(self, msg: str) -> None:
        """Log error message (highest priority)."""
        self._log(LOG_LEVEL_ERROR, "ERROR", msg)
    
    def set_level(self, level: str) -> None:
        """
        Change log level at runtime.
        
        Args:
            level: New log level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        """
        self.level = self.level_map.get(level.upper(), LOG_LEVEL_INFO)
    
    def get_level(self) -> str:
        """Get current log level as string."""
        for name, value in self.level_map.items():
            if value == self.level:
                return name
        return 'INFO'

