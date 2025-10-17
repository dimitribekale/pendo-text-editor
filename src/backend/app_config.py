import json
import os

CONFIG_FILE = "settings.json"

DEFAULT_CONFIG = {
    "font_family": "Consolas",
    "font_size": 12
}

def load_config():
    if not os.path.exists(CONFIG_FILE):
        return DEFAULT_CONFIG
    try:
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return DEFAULT_CONFIG

def save_config(config):
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=4)
    except IOError:
        print(f"Error: Could not save config to {CONFIG_FILE}")
