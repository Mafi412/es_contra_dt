# Environment wrapper abstract class for novelty search

from es_utilities.wrappers import EsEnvironmentWrapper

# --- "Abstract" environment wrapper ---

class NsEnvironmentWrapper(EsEnvironmentWrapper):
    def __init__(self, env, seed=None):
        super().__init__(env, seed)
        
    def get_behavior_characteristic(self): # Returns current behavior characteristic for the behavior of the agent since last reset.
        raise NotImplementedError()
