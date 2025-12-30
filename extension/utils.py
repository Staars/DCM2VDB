"""Utility functions and classes for DICOM import"""

# Simple logging wrapper since Blender's logging API isn't fully implemented yet
class SimpleLogger:
    def info(self, msg):
        print(f"INFO: {msg}")
    
    def debug(self, msg):
        print(f"DEBUG: {msg}")
    
    def warning(self, msg):
        print(f"WARNING: {msg}")
    
    def error(self, msg):
        print(f"ERROR: {msg}")
