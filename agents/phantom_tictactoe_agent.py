from agents.mlp_agent import MLPAgent
import chex


class PhantomTicTacToeAgent(MLPAgent):
    """Custom MLP agent for Phantom Tic-Tac-Toe environments."""
    
    def __init__(self, key: chex.PRNGKey):
        super().__init__(key, input_dim=9, output_dim=9)