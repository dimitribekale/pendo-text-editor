from backend.prediction_model import Prediction

class ModelManager:
    """
    Singleton class that manages a single shared AI model instance.
    This prevents loading the model multiple times (once per tab).
    """
    _instance = None
    _predictor = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance
    
    def get_predictor(self):
        """
        Returns the shared Prediction instance.
        Load it on the first access.
        """
        if self._predictor is None:
            print("Loading the AI model.")
            self._predictor = Prediction()
            print("AI model loaded successfully!")
        return self._predictor
    
    def is_loaded(self):
        """Check if the model has been loaded yet."""
        return self._predictor is not None