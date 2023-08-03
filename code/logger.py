import logging
import os

def get_log_file_path():
    config_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.ini")
    config = configparser.ConfigParser()
    config.read(config_file)
    log_file_path = config['LOG']['log_file_path']
    return log_file_path

# Set up logging configuration
log_file = get_log_file_path()
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')