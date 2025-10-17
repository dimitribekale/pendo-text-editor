import winreg

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

def get_system_theme():
    """Checks the Windows registry for the system theme.
    Returns 'dark' or 'light'. Defaults to 'light' on error or non-Windows OS.
    """
    try:
        # Registry key for theme settings
        key_path = r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize"
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path)
        
        # Value that indicates if apps should use the light theme (1) or dark theme (0)
        value, _ = winreg.QueryValueEx(key, "AppsUseLightTheme")
        winreg.CloseKey(key)
        
        return "light" if value == 1 else "dark"
    except (FileNotFoundError, OSError):
        # Default to light theme if key is not found or on other OSes
        return "light"
