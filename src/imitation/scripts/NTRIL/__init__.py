"""Noisy Trajectory Ranked Imitation Learning (NTRIL) package."""

# Import main components
try:
    from imitation.scripts.NTRIL.ntril import NTRILTrainer
    from imitation.scripts.NTRIL.noise_injection import NoiseInjector
    from imitation.scripts.NTRIL.robust_tube_mpc import RobustTubeMPC
    from imitation.scripts.NTRIL.ranked_dataset import RankedDatasetBuilder
    from imitation.scripts.NTRIL.demonstration_ranked_irl import DemonstrationRankedIRL
    
    __all__ = [
        "NTRILTrainer",
        "NoiseInjector", 
        "RobustTubeMPC",
        "RankedDatasetBuilder",
        "DemonstrationRankedIRL",
    ]
except ImportError as e:
    # Handle missing dependencies gracefully
    import warnings
    warnings.warn(f"Could not import NTRIL components: {e}")
    __all__ = []
