

# Define color palettes
DARK_THEME = {
    "bg": "#2E2E2E",
    "fg": "#FFFFFF",
    "widget_bg": "#3C3C3C",
    "widget_fg": "#FFFFFF",
    "text_bg": "#1E1E1E",
    "text_fg": "#D4D4D4",
    "cursor": "#FFFFFF",
    "select_bg": "#264F78",
    "line_number_bg": "#2E2E2E",
    "line_number_fg": "#858585",
}

LIGHT_THEME = {
    "bg": "#F0F0F0",
    "fg": "#000000",
    "widget_bg": "#FFFFFF",
    "widget_fg": "#000000",
    "text_bg": "#FFFFFF",
    "text_fg": "#000000",
    "cursor": "#000000",
    "select_bg": "#0078D7",
    "line_number_bg": "#F0F0F0",
    "line_number_fg": "#606366",
}

import sys
import subprocess


def get_system_theme():
    """
    Checks the system theme and "Dark" or "Light". Defaults to "Light" on error.
    """
    platform = sys.platform

    if platform == "win32":
        try:
            import winreg
            # Registry key for theme settings
            key_path = r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize"
            key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path)
            # Value that indicates if apps should use the light theme (1) or dark theme (0)
            value, _ = winreg.QueryValueEx(key, "AppsUseLightTheme")
            winreg.CloseKey(key)
            return "light" if value == 1 else "dark"
        
        except (FileNotFoundError, OSError, ModuleNotFoundError):
            return "light"
    
    elif platform == "darwin":
        # MacOs
        try:
            result = subprocess.run(
                ["defaults", "read", "-g", "AppleInterfaceStyle"],
                capture_output=True,
                text=True,
                timeout=2
            )
            return "dark" if result.stdout.strip().lower() == "dark" else "light"
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            return "light"
    
    elif platform.startswith("linux"):
        try:
            result = subprocess.run(
                ["getsettings", "get", "org.gnome.desktop.interface", "gtk-theme"],
                capture_output=True,
                text=True,
                timeout=2
            )
            theme_name = result.stdout.strip().lower()
            return "dark" if "dark" in theme_name else "light"
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            return "light"
        
    return "light"
