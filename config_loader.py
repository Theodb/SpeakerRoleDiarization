# config_loader.py
import os
import configparser

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, 'config.ini')

config = configparser.ConfigParser()
config.read(CONFIG_PATH)