from agents.base_agent import BaseAgent
from agents.tictactoe import TicTacToeAgent
from agents.battleship import BattleShipAgent
from agents.kuhn_poker import KuhnPokerAgent
from agents.leduc_poker import LeducPokerAgent
from agents.liar_dice import LiarDiceAgent
from agents.poker import HeadUpPokerAgent
from agents.mixture_agent import MixtureAgent
from agents.random_agent import RandomAgent
from agents.mlp_agent import MLPAgent
from agents.phantom_tictactoe_agent import PhantomTicTacToeAgent
from agents.dark_hex3_agent import DarkHex3Agent
from enum import Enum
from typing import Union
import chex


class RegisteredAgent(Enum):
    TIC_TAC_TOE = "tic_tac_toe"
    BATTLE_SHIP = "battle_ship"
    KUHN_POKER = "kuhn_poker"
    LEDUC_POKER = "leduc_poker"
    LIAR_DICE = "liar_dice"
    HEAD_UP_POKER = "head_up_poker"
    PHANTOM_TICTACTOE_CLASSIC = "phantom_tic_tac_toe_classic"
    PHANTOM_TICTACTOE_ABRUPT = "phantom_tic_tac_toe_abrupt"
    DARK_HEX3_CLASSIC = "dark_hex3_classic"
    DARK_HEX3_ABRUPT = "dark_hex3_abrupt"


def create_agent(agent_name: Union[RegisteredAgent, str], key: chex.PRNGKey) -> BaseAgent:
    """Create an agent instance based on the registered agent name.
    
    Args:
        agent_name: The registered agent type to create (enum or string)
        rngs: JAX random number generator state for initialization
        
    Returns:
        An instance of the specified agent type
        
    Raises:
        ValueError: If the agent name is not recognized
    """
    # Convert string to enum if needed
    if isinstance(agent_name, str):
        try:
            agent_name = RegisteredAgent(agent_name)
        except ValueError:
            raise ValueError(f"Unknown agent: {agent_name}")
        
    if agent_name == RegisteredAgent.TIC_TAC_TOE:
        return TicTacToeAgent(key)
    elif agent_name == RegisteredAgent.BATTLE_SHIP:
        return BattleShipAgent(key)
    elif agent_name == RegisteredAgent.KUHN_POKER:
        return KuhnPokerAgent(key)
    elif agent_name == RegisteredAgent.LIAR_DICE:
        return LiarDiceAgent(key)
    elif agent_name == RegisteredAgent.HEAD_UP_POKER:
        return HeadUpPokerAgent(key)
    elif agent_name == RegisteredAgent.LEDUC_POKER:
        return LeducPokerAgent(key)
    elif agent_name == RegisteredAgent.PHANTOM_TICTACTOE_CLASSIC:
        return PhantomTicTacToeAgent(key)
    elif agent_name == RegisteredAgent.PHANTOM_TICTACTOE_ABRUPT:
        return PhantomTicTacToeAgent(key)
    elif agent_name == RegisteredAgent.DARK_HEX3_CLASSIC:
        return DarkHex3Agent(key)
    elif agent_name == RegisteredAgent.DARK_HEX3_ABRUPT:
        return DarkHex3Agent(key)
    else:
        raise ValueError(f"Unknown agent: {agent_name}")
    

__all__ = ['BaseAgent', 'RegisteredAgent', 'create_agent', 'MixtureAgent', 'RandomAgent']