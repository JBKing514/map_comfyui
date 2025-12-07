from .map_nodes import MAP_Trajectory_Sampler

NODE_CLASS_MAPPINGS = {
    "MAP_Trajectory_Sampler": MAP_Trajectory_Sampler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MAP_Trajectory_Sampler": "MAP Trajectory Sampler (PCA)"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]