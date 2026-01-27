from proctoring_ml_module.engines.proctoring_engine import ProctoringEngine

# Public API Facade

def create_engine(config_path=None):
    """
    Factory to create a new Proctoring Monitoring Engine.
    Args:
        config_path (str, optional): Path to config.yaml.
    Returns:
        ProctoringEngine instance.
    """
    return ProctoringEngine(config_path)

# If we wanted to enforce a singleton or session management here, we could.
# For now, we expose the direct engine creation.
