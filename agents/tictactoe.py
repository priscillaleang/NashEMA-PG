from typing import Tuple
from functools import partial
from flax import nnx
from agents import BaseAgent
from agents.utils import layer_init
import jax
import jax.numpy as jnp
import chex
import distrax

EMB_SIZE = 4
CONV_CHANNELS = 16

class TicTacToeAgent(BaseAgent):
    
    def __init__(self, key: chex.PRNGKey):
        rngs = nnx.Rngs(key)

        # Separate embedding layers for policy and critic
        self.policy_board_embedding = nnx.Embed(3, EMB_SIZE, rngs=rngs)
        self.critic_board_embedding = nnx.Embed(3, EMB_SIZE, rngs=rngs)
        
        # Separate convolutional layers for policy
        self.policy_conv_layers = nnx.Sequential(
            nnx.Conv(EMB_SIZE, CONV_CHANNELS, kernel_size=(3, 3), padding="SAME", rngs=rngs),
            nnx.relu,
            nnx.Conv(CONV_CHANNELS, CONV_CHANNELS, kernel_size=(3, 3), padding="SAME", rngs=rngs),
            nnx.relu
        )
        
        # Separate convolutional layers for critic
        self.critic_conv_layers = nnx.Sequential(
            nnx.Conv(EMB_SIZE, CONV_CHANNELS, kernel_size=(3, 3), padding="SAME", rngs=rngs),
            nnx.relu,
            nnx.Conv(CONV_CHANNELS, CONV_CHANNELS, kernel_size=(3, 3), padding="SAME", rngs=rngs),
            nnx.relu
        )
        
        # Policy network
        self._policy_head = nnx.Linear(in_features=CONV_CHANNELS, out_features=9, rngs=rngs)
        self.policy_layers = nnx.Sequential(
            lambda x: jnp.mean(x, axis=(1, 2)),  # Global average pooling: (batch, 3, 3, 16) -> (batch, 16)
            nnx.Linear(in_features=CONV_CHANNELS, out_features=CONV_CHANNELS, rngs=rngs),
            nnx.relu,
            self._policy_head
        )
        
        # Critic network  
        self.critic_layers = nnx.Sequential(
            lambda x: jnp.mean(x, axis=(1, 2)),  # Global average pooling: (batch, 3, 3, 16) -> (batch, 16)
            nnx.Linear(in_features=CONV_CHANNELS, out_features=CONV_CHANNELS, rngs=rngs),
            nnx.relu,
            nnx.Linear(in_features=CONV_CHANNELS, out_features=1, rngs=rngs)
        )

        # init modules
        layer_init(self, rngs.param())
        # init policy head
        layer_init(self._policy_head, rngs.param(), std=0.01)
        
    def _process_input_policy(self, observations: chex.Array) -> chex.Array:
        """Process input observations through policy embedding and conv layers.
        
        Args:
            observations: Board observations of shape (batch_size, 3, 3) with values {-1, 0, 1}
            
        Returns:
            Processed features of shape (batch_size, 3, 3, 128)
        """
        # Convert {-1, 0, 1} to {0, 1, 2} for embedding
        board_indices = (observations + 1).astype(jnp.int32)
        
        # Embed board values: (batch_size, 3, 3) -> (batch_size, 3, 3, 32)
        board_embedded = self.policy_board_embedding(board_indices)
        
        # Apply policy convolutional layers: (batch_size, 3, 3, 32) -> (batch_size, 3, 3, 128)
        features = self.policy_conv_layers(board_embedded)
        
        return features
    
    def _process_input_critic(self, observations: chex.Array) -> chex.Array:
        """Process input observations through critic embedding and conv layers.
        
        Args:
            observations: Board observations of shape (batch_size, 3, 3) with values {-1, 0, 1}
            
        Returns:
            Processed features of shape (batch_size, 3, 3, 128)
        """
        # Convert {-1, 0, 1} to {0, 1, 2} for embedding
        board_indices = (observations + 1).astype(jnp.int32)
        
        # Embed board values: (batch_size, 3, 3) -> (batch_size, 3, 3, 32)
        board_embedded = self.critic_board_embedding(board_indices)
        
        # Apply critic convolutional layers: (batch_size, 3, 3, 32) -> (batch_size, 3, 3, 128)
        features = self.critic_conv_layers(board_embedded)
        
        return features

    @partial(jax.jit, static_argnums=0)
    def get_value(self, observations: chex.Array) -> chex.Array:
        """Compute state value using the value network.
        
        Args:
            observations: Observation tensor of shape (batch_size, 3, 3)
            
        Returns:
            Value estimates of shape (batch_size,)
        """
        chex.assert_rank(observations, 3)  # (batch_size, 3, 3)
        chex.assert_shape(observations, (observations.shape[0], 3, 3))

        features = self._process_input_critic(observations)
        return self.critic_layers(features).squeeze(-1)
    
    @partial(jax.jit, static_argnums=0)
    def get_action(self, observations: chex.Array, key: chex.PRNGKey, action_masks: chex.Array = None) -> chex.Array:
        """Sample action from policy network.
        
        Args:
            observations (batch_size, *observation_shape): Observation tensor
            key (): JAX random key for sampling
            action_masks (batch_size, num_actions): Binary mask for valid actions
            
        Returns:
            Sampled actions of shape (batch_size, )
        """
        chex.assert_rank(key, 0)  # jax.random.key() has rank 0

        dist = self.get_action_distribution(observations, action_masks)
        actions = dist.sample(seed=key)

        return actions
    
    @partial(jax.jit, static_argnums=0)
    def get_action_and_value(
            self, observations: chex.Array, key: chex.PRNGKey, action_masks: chex.Array = None
        ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """Sample action and compute log probability and value simultaneously.
        
        Args:
            observations: Observation tensor of shape (batch_size, 3, 3)
            key: JAX random key for sampling
            action_masks: Binary mask for valid actions of shape (batch_size, 9)
        
        Returns:
            Tuple of (action, log_prob, value) arrays of shape (batch_size,)
        """
        chex.assert_rank(observations, 3)  # (batch_size, 3, 3)
        chex.assert_shape(observations, (observations.shape[0], 3, 3))
        chex.assert_rank(key, 0)  # jax.random.key() has rank 0

        critic_features = self._process_input_critic(observations)
        
        dist = self.get_action_distribution(observations, action_masks)
        actions, log_probs = dist.sample_and_log_prob(seed=key)
        values = self.critic_layers(critic_features).squeeze(-1)

        return actions, log_probs, values
    
    @partial(jax.jit, static_argnums=0)
    def get_action_distribution(
        self, observations: chex.Array, action_masks: chex.Array = None
    ) -> distrax.Distribution:
        """Get action distribution
        
        Args:
            observations: Observation tensor of shape (batch_size, 3, 3)
            action_masks: Binary mask for valid actions of shape (batch_size, 9)
            
        Returns:
            distrax.Distribution of shape (batch_size, )
        """
        chex.assert_rank(observations, 3)  # (batch_size, 3, 3)
        chex.assert_shape(observations, (observations.shape[0], 3, 3))

        features = self._process_input_policy(observations)
        logits = self.policy_layers(features)
        
        if action_masks is not None:
            logits = jnp.where(action_masks, logits, -jnp.inf)
        
        dist = distrax.Categorical(logits=logits)
        return dist

