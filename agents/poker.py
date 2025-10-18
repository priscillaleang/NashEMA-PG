from typing import Tuple
from functools import partial
from flax import nnx
from agents import BaseAgent
from agents.utils import layer_init
import jax.numpy as jnp
import jax
import chex
import distrax

from agents.blocks.transformer import TransformerBlock
import envs.mytypes as env_types
from envs.head_up_poker import MAX_HISTORY_LENGTH, STARTING_STACK, NUM_ACTIONS

CARD_EMB_SIZE = 128
MONEY_EMB_SIZE = 16
STREET_EMB_SIZE = 16
HIST_EMB_SIZE = CARD_EMB_SIZE + 2 * STREET_EMB_SIZE + 6 * MONEY_EMB_SIZE

TRANFORMER_MLP_DIM = 256 # the MLP dimension of transformer
FINAL_MLP_DIM = 256 # the MLP dimension of policy/critic mlp

NUM_CARD_LAYER = 1 # number of transformer block for card encoder
NUM_HIST_LAYER = 1 # number of transformer block for History feature encoder

class FeatureExtractor(nnx.Module):

    def __init__(self, key: chex.PRNGKey):
        rngs = nnx.Rngs(key)

        self.token_type_embedding = nnx.Embed(3, HIST_EMB_SIZE, rngs=rngs) # (-1=padding, 0=player, 1=oppponent)
        self.card_embedding = nnx.Embed(53, CARD_EMB_SIZE, rngs=rngs) # card [-1 ~ 51]
        self.card_type_embedding = nnx.Embed(2, CARD_EMB_SIZE, rngs=rngs) # (0=hole, 1=board) card
        self.street_embedding = nnx.Embed(4, STREET_EMB_SIZE, rngs=rngs) # street [0 ~ 3]
        self.money_embedding = nnx.Linear(1, MONEY_EMB_SIZE, rngs=rngs)
        self.hist_position_embedding = nnx.Embed(MAX_HISTORY_LENGTH, HIST_EMB_SIZE, rngs=rngs)

        self.card_encoder_blocks = nnx.Sequential(*[
            TransformerBlock(features=CARD_EMB_SIZE, num_heads=8, mlp_dim=TRANFORMER_MLP_DIM, rngs=rngs)
            for _ in range(NUM_CARD_LAYER)
        ])
        self.history_encoder_blocks = nnx.Sequential(*[
            TransformerBlock(features=HIST_EMB_SIZE, num_heads=8, mlp_dim=TRANFORMER_MLP_DIM, rngs=rngs)
            for _ in range(NUM_HIST_LAYER)
        ])

    def __call__(self, obs: env_types.Observation):
        hole_cards = obs['hole_cards']
        board_cards = obs['board_cards']
        pot_size = obs['pot_size']
        street = obs['street']
        stacks = obs['stacks']
        bets = obs['bets']
        action_history = obs['action_history']
        # we don't use `bet_amount`
        
        """process card feature"""
        cards_emb = self.card_embedding(jnp.concatenate([hole_cards + 1, board_cards + 1], axis=1)) # (batch, 7, CARD_EMB_SIZE)
        hole_type = jnp.zeros((hole_cards.shape[0], 2), dtype=jnp.int32)
        board_type = jnp.ones((board_cards.shape[0], 5), dtype=jnp.int32)
        cards_type_emb = self.card_type_embedding(jnp.concatenate([hole_type, board_type], axis=1)) # (batch, 7, CARD_EMB_SIZE)

        # position encoding
        cards_emb = cards_emb + cards_type_emb
        card_feature = self.card_encoder_blocks(cards_emb) # (batch, 7, CARD_EMB_SIZE)
        card_feature = jnp.mean(card_feature, axis=1) # (batch, CARD_EMB_SIZE)
        
        """process meta features"""
        # helper function
        norm_money_fc = lambda x: (x.astype(jnp.float32) / float(STARTING_STACK))
        street_emb = self.street_embedding(street) # (batch, STREET_EMB_SIZE)
        pot_size_emb = self.money_embedding(norm_money_fc(pot_size)[..., None]) # (batch, MONEY_EMB_SIZE)
        stacks_emb = self.money_embedding(norm_money_fc(stacks)[..., None]) # (batch, 2, MONEY_EMB_SIZE)
        bets_emb = self.money_embedding(norm_money_fc(bets)[..., None]) # (batch, 2, MONEY_EMB_SIZE)

        """process action history features"""
        hist_token_type_emb = self.token_type_embedding(action_history[:,:,0] + 1) # (batch, MAX_HISTORY_LENGTH, HIST_EMB_SIZE)
        hist_street_emb = self.street_embedding(action_history[:,:,1]) # (batch, MAX_HISTORY_LENGTH, STREET_EMB_SIZE)
        hist_money_emb = self.money_embedding(norm_money_fc(action_history[:,:,3:4])) # (batch, MAX_HISTORY_LENGTH, MONEY_EMB_SIZE)

        """fusion features"""
        batch_size = card_feature.shape[0]
        hist_pos_emb = self.hist_position_embedding(jnp.arange(MAX_HISTORY_LENGTH)) # (MAX_HISTORY_LENGTH, HIST_EMB_SIZE)

        # Broadcast game-level features to match history length
        card_feature_broadcast = jnp.broadcast_to(
            card_feature[:, None, :], (batch_size, MAX_HISTORY_LENGTH, CARD_EMB_SIZE)
        ) # (batch, MAX_HISTORY_LENGTH, CARD_EMB_SIZE)
        
        street_emb_broadcast = jnp.broadcast_to(
            street_emb[:, None, :], (batch_size, MAX_HISTORY_LENGTH, STREET_EMB_SIZE)
        ) # (batch, MAX_HISTORY_LENGTH, STREET_EMB_SIZE)
        
        pot_size_emb_broadcast = jnp.broadcast_to(
            pot_size_emb[:, None, :], (batch_size, MAX_HISTORY_LENGTH, MONEY_EMB_SIZE)
        ) # (batch, MAX_HISTORY_LENGTH, MONEY_EMB_SIZE)
        
        stacks_emb_broadcast = jnp.broadcast_to(
            stacks_emb[:, None, :, :], (batch_size, MAX_HISTORY_LENGTH, 2, MONEY_EMB_SIZE)
        ).reshape(batch_size, MAX_HISTORY_LENGTH, 2 * MONEY_EMB_SIZE) # (batch, MAX_HISTORY_LENGTH, 2*MONEY_EMB_SIZE)
        
        bets_emb_broadcast = jnp.broadcast_to(
            bets_emb[:, None, :, :], (batch_size, MAX_HISTORY_LENGTH, 2, MONEY_EMB_SIZE)
        ).reshape(batch_size, MAX_HISTORY_LENGTH, 2 * MONEY_EMB_SIZE) # (batch, MAX_HISTORY_LENGTH, 2*MONEY_EMB_SIZE)

        # Concatenate all features for each history token
        # Each token: card_feature + street_emb + pot_size_emb + stacks_emb[2] + bets_emb[2] + hist_street_emb + hist_money_emb
        hist_features = jnp.concatenate([
            card_feature_broadcast,    # CARD_EMB_SIZE
            street_emb_broadcast,      # STREET_EMB_SIZE  
            pot_size_emb_broadcast,    # MONEY_EMB_SIZE
            stacks_emb_broadcast,      # 2 * MONEY_EMB_SIZE
            bets_emb_broadcast,        # 2 * MONEY_EMB_SIZE
            hist_street_emb,           # STREET_EMB_SIZE
            hist_money_emb             # MONEY_EMB_SIZE
        ], axis=-1) # (batch, MAX_HISTORY_LENGTH, HIST_EMB_SIZE)

        # Add positional and token type embeddings
        hist_features = hist_features + hist_token_type_emb + hist_pos_emb[None, :, :] # (batch, MAX_HISTORY_LENGTH, HIST_EMB_SIZE)

        """process by main transformer block"""
        hist_features = self.history_encoder_blocks(hist_features) # (batch, MAX_HISTORY_LENGTH, HIST_EMB_SIZE)
        hist_features = jnp.mean(hist_features, axis=1) # (batch, HIST_EMB_SIZE)

        return hist_features 

        
class HeadUpPokerAgent(BaseAgent):
    """Agent for Liar's Dice using 1D convolution with separate policy and critic networks."""
    
    def __init__(self, key: chex.PRNGKey):
        key,subkey1, subkey2 = jax.random.split(key, 3)
        rngs = nnx.Rngs(key)
        
        self.policy_feature_extractor = FeatureExtractor(subkey1)
        self.critic_feature_extractor = FeatureExtractor(subkey2)

        self.policy_head = nnx.Linear(in_features=FINAL_MLP_DIM, out_features=NUM_ACTIONS, rngs=rngs)
        self.policy_mlp = nnx.Sequential(
            nnx.Linear(in_features=HIST_EMB_SIZE, out_features=FINAL_MLP_DIM, rngs=rngs),
            nnx.relu,
            self.policy_head
        )

        self.critic_mlp = nnx.Sequential(
            nnx.Linear(in_features=HIST_EMB_SIZE, out_features=FINAL_MLP_DIM, rngs=rngs),
            nnx.relu,
            nnx.Linear(in_features=FINAL_MLP_DIM, out_features=1, rngs=rngs)
        )
        
        # Initialize all layers
        layer_init(self, rngs.param())
        # Initialize policy head with smaller std
        layer_init(self.policy_head, rngs.param(), std=0.01)
    
    @partial(jax.jit, static_argnums=0)
    def get_value(self, observations: env_types.Observation) -> chex.Array:
        """Compute state value using the critic network."""

        features = self.critic_feature_extractor(observations)
        values = self.critic_mlp(features).squeeze(-1)

        return values
        
    
    @partial(jax.jit, static_argnums=0)
    def get_action(self, observations: env_types.Observation, key: chex.PRNGKey, action_masks: chex.Array = None) -> chex.Array:
        """Sample action from policy network."""
        dist = self.get_action_distribution(observations, action_masks)
        actions = dist.sample(seed=key)
        return actions
    
    @partial(jax.jit, static_argnums=0)
    def get_action_and_value(
        self, observations: env_types.Observation, key: chex.PRNGKey, action_masks: chex.Array = None
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """Sample action and compute log probability and value simultaneously."""

        # Critic processing
        values = self.get_value(observations)

        # Get action distribution and sample
        dist = self.get_action_distribution(observations, action_masks)
        actions, log_probs = dist.sample_and_log_prob(seed=key)
        
        return actions, log_probs, values
    
    @partial(jax.jit, static_argnums=0)
    def get_action_distribution(
        self, observations: env_types.Observation, action_masks: chex.Array = None
    ) -> distrax.Distribution:
        """Get action distribution from policy network."""

        features = self.policy_feature_extractor(observations)
        logits = self.policy_mlp(features)

        # Apply action mask if provided
        if action_masks is not None:
            logits = jnp.where(action_masks, logits, -jnp.inf)
        
        return distrax.Categorical(logits=logits)
        