import yaml
from pathlib import Path


CONFIG_FILE = Path(__file__).parent.parent.parent / "config.yaml"

def _deep_merge(default, user):
    """Recursively merge user config into default config."""
    result = default.copy()
    for key, value in user.item():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result

def load_config():
    """Load configuration from YAML file."""
    if not CONFIG_FILE.exists():
        print(f"Config file not found at {CONFIG_FILE}. Using defaults from config.yaml template.")
        return None
    
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            return config if config else {}
        
    except (yaml.YAMLError, IOError) as e:
        print(f"Error loading config: {e}. Using defaults.")
        return {}

def save_config(config):
    """Save configuration to YAML file."""
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    except IOError as e:
        print(f"Error: Could not save config tp {CONFIG_FILE}: {e}")

def get_value(config, path, default=None):
    """
    Get nested config value using dot notation.
    Example: get_value(config, 'editor.font_family', 'Consolas')
    """
    keys = path.split(".")
    value = config
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return value