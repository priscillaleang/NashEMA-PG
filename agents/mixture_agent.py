from pathlib import Path
from typing import List, Tuple
from flax import nnx
import chex
import abc
import distrax
import jax.numpy as jnp
import jax

import envs.mytypes as env_types
from agents import BaseAgent

class MixtureAgent(BaseAgent):
    """
    JAX-compatible mixture agents using stacked parameters.
    """

    def __init__(self, key: chex.PRNGKey, agents: List[BaseAgent] = None, mixture_logits: chex.Array = None, mixture_probs: chex.Array = None):
        """
        Initialize mixture agent with batched agents of same type/shape.
        
        Args:
            key: JAX random key (required for BaseAgent interface)
            agents: List of agents that are guaranteed to be same type/shape
            mixture_logits: Logits for mixture distribution over agents (mutually exclusive with mixture_probs)
            mixture_probs: Probabilities for mixture distribution over agents (mutually exclusive with mixture_logits)
        """
        assert agents is not None, "must specify agents"
        assert len(agents) > 0, "must provide at least one agent"
        
        # Ensure exactly one of mixture_logits or mixture_probs is provided
        if mixture_logits is not None and mixture_probs is not None:
            raise ValueError("Cannot specify both mixture_logits and mixture_probs")
        if mixture_logits is None and mixture_probs is None:
            raise ValueError("Must specify either mixture_logits or mixture_probs")
        
        # Store the mixture parameters
        if mixture_logits is not None:
            assert len(agents) == len(mixture_logits), "shape of agents and mixture_logits doesn't fit"
            self.mixture_logits = mixture_logits
            self.mixture_probs = jax.nn.softmax(mixture_logits)
        else:
            assert len(agents) == len(mixture_probs), "shape of agents and mixture_probs doesn't fit"
            self.mixture_probs = mixture_probs
            self.mixture_logits = jnp.log(mixture_probs)

        self.num_agents = len(agents)
        self.agents = agents
        
        # Extract pytree structure from agents and stack their parameters
        self._setup_stacked_agents(agents)

    def _setup_stacked_agents(self, agents: List[BaseAgent]):
        """Extract pytree structure and stack agent parameters."""
        # Split first agent to get graphdef and state structure
        graphdef, first_state = nnx.split(agents[0])
        
        # Verify all agents have the same structure
        for i, agent in enumerate(agents[1:], 1):
            _, agent_state = nnx.split(agent)
            # Check that graphdefs are equivalent (same structure)
            if not jax.tree.structure(first_state) == jax.tree.structure(agent_state):
                raise ValueError(f"Agent {i} has different pytree structure than agent 0")
        
        # Store the common graphdef
        self.agent_graphdef = graphdef
        
        # Stack all agent states
        all_states = [nnx.split(agent)[1] for agent in agents]
        self.stacked_states = jax.tree.map(lambda *states: jnp.stack(states, axis=0), *all_states)

    def _reconstruct_agent(self, agent_idx: int) -> BaseAgent:
        """Reconstruct an agent from stacked parameters by index."""
        # Extract state for specific agent
        agent_state = jax.tree.map(lambda x: x[agent_idx], self.stacked_states)
        
        # Reconstruct agent using stored graphdef and extracted state
        return nnx.merge(self.agent_graphdef, agent_state)

    def get_value(self, observations: env_types.Observation) -> chex.Array:
        """Compute state value using the value network.
        
        Args:
            state: Observation tensor of shape (batch_size, *observation_shape)
            
        Returns:
            Value estimates of shape (batch_size,)
        """
        return self.get_value_by_index(observations, 0)
    
    def get_action(self, observations: env_types.Observation, key: chex.PRNGKey, action_masks: chex.Array = None) -> env_types.Action:
        """Sample action from policy network.
        
        Args:
            observations (batch_size, *observation_shape): Observation tensor
            key (): JAX random key for sampling
            action_masks (batch_size, num_actions): Binary mask for valid actions
            
        Returns:
            Sampled actions of shape (batch_size, )
        """
        return self.get_action_and_value_by_index(observations, key, action_masks, 0)
    
    def get_action_and_value(
            self, observations: env_types.Observation, key: chex.PRNGKey, action_masks: chex.Array = None
        ) -> Tuple[env_types.Action, chex.Array, chex.Array]:
        """Sample action and compute log probability and value simultaneously.
        
        Args:
            observations (batch_size, *observation_shape): Observation tensor
            key (): JAX random key for sampling
            action_masks (batch_size, num_actions): Binary mask for valid actions
        
        Returns:
            Tuple of (action, log_prob, value) arrays of shape (batch_size,)
        """
        return self.get_action_and_value_by_index(observations, key, action_masks, 0)
    
    def get_action_distribution(
        self, observations: env_types.Observation, action_masks: chex.Array = None
    ) -> distrax.Distribution:
        """Get action distribution.
        
        Args:
            observations: Observation tensor of shape (batch_size, *observation_shape)
            action_masks (batch_size, num_actions): Binary mask for valid actions
            
        Returns:
            distrax.Distribution of shape (batch_size,)
        """
        return self.get_action_distribution_by_index(observations, action_masks, 0)
    
    def get_action_distribution_by_index(
        self, observations: env_types.Observation, action_masks: chex.Array, agent_idx: int
    ) -> distrax.Distribution:
        """Get action distribution from a specific agent by index."""
        agent = self._reconstruct_agent(agent_idx)
        return agent.get_action_distribution(observations, action_masks)
    
    def get_action_and_value_by_index(
        self, observations: env_types.Observation, key: chex.PRNGKey, 
        action_masks: chex.Array, agent_idx: int
    ) -> Tuple[env_types.Action, chex.Array, chex.Array]:
        """Get action and value from a specific agent by index."""
        agent = self._reconstruct_agent(agent_idx)
        return agent.get_action_and_value(observations, key, action_masks)
    
    def get_value_by_index(self, observations: env_types.Observation, agent_idx: int) -> chex.Array:
        """Compute state value using a specific agent by index."""
        agent = self._reconstruct_agent(agent_idx)
        return agent.get_value(observations)
    
    def maybe_resample(self, active_agent_idx: int, episode_done: bool, key: chex.PRNGKey) -> Tuple[int, chex.PRNGKey]:
        """Resample active agent index when episode ends."""
        key, sample_key = jax.random.split(key)
        new_idx = jax.random.categorical(sample_key, self.mixture_logits)
        
        # Use jnp.where to conditionally return new or old index
        final_idx = jnp.where(episode_done, new_idx, active_agent_idx)
        return final_idx, key