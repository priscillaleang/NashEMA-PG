from agents.mlp_agent import MLPAgent
import chex


class DarkHex3Agent(MLPAgent):
    """Custom MLP agent for Dark Hex3 environments."""
    
    def __init__(self, key: chex.PRNGKey):
        super().__init__(key, input_dim=9, output_dim=9)