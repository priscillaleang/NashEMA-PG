"""Agent classes for ELO computation."""

from typing import List
import numpy as np
import jax.numpy as jnp
import jax

from agents.base_agent import BaseAgent
from scripts.compute_elo.config import INITIAL_ELO


def batch_observation(observation):
    """Add a batch dimension to observation pytree for agents that expect batched input."""
    return jax.tree.map(lambda x: jnp.expand_dims(x, axis=0), observation)


class MixtureAgent(BaseAgent):
    def __init__(self, agents: List[BaseAgent], prob: List[float]):
        # Check if probabilities sum to 1
        prob_sum = sum(prob)
        if abs(prob_sum - 1.0) > 1e-6:
            print(f"Warning: probabilities sum to {prob_sum:.6f}, not 1.0. Normalizing...")
            prob = [p / prob_sum for p in prob]
        
        if len(agents) != len(prob):
            print(f"Warning: number of agent {len(agents)} should match number of prob {len(prob)}")

        self.agents = agents
        self.prob = np.array(prob)

    def sample_agent(self) -> BaseAgent:
        # Ensure probabilities sum to 1 (handle floating point precision issues)
        prob_normalized = self.prob / self.prob.sum()
        sampled_idx = np.random.choice(len(self.prob), p=prob_normalized)
        return self.agents[sampled_idx]
    
    def get_value(self, observations):
        raise TypeError("Should not call elo mixture agent methods")

    def get_action(self, observations, key, action_masks=None):
        raise TypeError("Should not call elo mixture agent methods")

    def get_action_and_value(self, observations, key, action_masks=None):
        raise TypeError("Should not call elo mixture agent methods")

    def get_action_distribution(self, observations, action_masks=None):
        raise TypeError("Should not call elo mixture agent methods")


class Player:
    def __init__(self, agent: BaseAgent, family: str, step: int, elo: float = INITIAL_ELO):
        self.agent = agent
        self.family = family  # e.g., "nash_pg_mag0.0_kl", "fsp", "psro"
        self.step = step
        self.elo = elo
    
    def __repr__(self):
        return f"Player({self.family}, step={self.step}, elo={self.elo:.1f})"


def get_playing_agent(player: Player) -> BaseAgent:
    """Get the actual agent to play (sample if mixture agent)"""
    agent = player.agent
    if isinstance(agent, MixtureAgent):
        return agent.sample_agent()
    return agent