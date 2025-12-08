"""
Editor Frame Factory for Pendo Text Editor.

Encapsulates the creation and configuration of EditorFrame instances,
reducing coupling between FileManager and the application instance.
"""

from ..editor_frame import EditorFrame


class EditorFrameFactory:
    """Factory for creating configured editor frames."""

    def __init__(self, notebook, theme, change_callback, predictor_provider,
                 context_menu_binder, settings_applicator, config):
        """
        Initialize factory with required dependencies.

        Args:
            notebook: ttk.Notebook widget to add frames to
            theme: Theme dictionary
            change_callback: Callback for text changes
            predictor_provider: Callable that returns a predictor instance
            context_menu_binder: Callable to bind context menu to editor frame
            settings_applicator: Callable to apply settings to editor frame
            config: Application configuration dictionary
        """
        self.notebook = notebook
        self.theme = theme
        self.change_callback = change_callback
        self.predictor_provider = predictor_provider
        self.context_menu_binder = context_menu_binder
        self.settings_applicator = settings_applicator
        self.config = config

    def create_editor_frame(self):
        """
        Create a new editor frame with standard configuration.

        Returns:
            EditorFrame: Configured editor frame ready to use
        """
        editor_frame = EditorFrame(
            self.notebook,
            theme=self.theme,
            change_callback=self.change_callback,
            predictor=self.predictor_provider()
        )

        # Apply standard post-creation configuration
        self.context_menu_binder(editor_frame)
        self.settings_applicator(self.config)

        return editor_frame
