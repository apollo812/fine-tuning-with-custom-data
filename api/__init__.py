import logging.config
from pathlib import Path

# Configure the logging settings using the contents of the "logging.conf" file.
# The path to the configuration file is constructed using the parent path of the current file.
# Existing loggers are not disabled to allow the new configuration to be applied without affecting existing loggers.
logging.config.fileConfig(Path(__file__).parent / "logging.conf", disable_existing_loggers=False)
