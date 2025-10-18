from typing import Any, Literal, Tuple
from functools import partial

import chex
import jax
from flax import nnx
import jax.numpy as jnp
import numpy as np

from agents import MixtureAgent, BaseAgent
from envs.mytypes import BaseEnv, EnvState, TimeStep


def update_meta_agent(
        meta_agent: MixtureAgent,
        new_agent: BaseAgent,
        key: chex.PRNGKey,
        env: BaseEnv,
        num_steps: int = 1_000,
        num_envs: int = 128,
        solver: Literal['nash', 'uniform'] = 'nash',
        num_solver_iteration = 10_000,
) -> MixtureAgent:
    """
    Update the meta agent by adding a new agent and recomputing the mixture weights.
    
    Args:
        meta_agent: Current mixture agent containing existing agents and their weights
        new_agent: New agent to add to the mixture
        key: JAX random key for initialization
        env: Environment instance for evaluating agent interactions
        num_steps: Number of simulation steps for meta-game evaluation
        num_envs: Number of parallel environments for evaluation
        solver: Meta-game solver method ('uniform' for equal weights, 'nash' for Nash equilibrium)
        num_solver_iteration: Number of iterations for fictitious play when using Nash solver
        
    Returns:
        Updated MixtureAgent with new agent added and recomputed mixture weights
    """

    if solver == 'uniform':
        all_agents = meta_agent.agents + [new_agent]
        num_agents = len(all_agents)
        mixture_logits = jnp.ones(shape=(num_agents, ), dtype=jnp.float32)

        return MixtureAgent(key, agents=all_agents, mixture_logits=mixture_logits)
    
    elif solver == 'nash':
        all_agents = meta_agent.agents + [new_agent]
        num_agents = len(all_agents)

        # run competition to create meta game matrix
        payoff_matrix = np.zeros((num_agents, num_agents), dtype=np.float32)

        for i in range(num_agents):
             for j in range(i + 1, num_agents):
                key, subkey = jax.random.split(key)
                avg_return = compute_avg_return(all_agents[i], all_agents[j], subkey, env, num_envs, num_steps)
                payoff_matrix[i, j] = avg_return
                payoff_matrix[j, i] = -avg_return

        # solve the game using fiticious play
        mixture_probs = compute_meta_strategy(payoff_matrix, num_iteration=num_solver_iteration)

        return MixtureAgent(key, agents=all_agents, mixture_probs=mixture_probs)

@partial(nnx.jit, static_argnames=('env', 'num_envs', 'num_steps'))
def compute_avg_return(
    agent_i: BaseAgent,
    agent_j: BaseAgent,
    key: chex.PRNGKey,
    env: BaseEnv,
    num_envs: int,
    num_steps: int
) -> chex.Numeric:
    """calculate the expected payoff of agent_i (playing against agent_j)"""

    def collect_one_env_step(carry: Tuple[TimeStep, EnvState, chex.PRNGKey, chex.Numeric, chex.Numeric, chex.Numeric] , _: Any):
            """step env for a single env, i.e., no batch dimesions"""

            last_timestep, env_state, key, acc_reward, total_return, num_eps  = carry
            key, act_key = jax.random.split(key)

            # add batch dimesion for input to agent
            last_timestep: TimeStep = jax.tree.map(lambda x: jnp.expand_dims(x, axis=0), last_timestep)
            
            # agent act
            action_i = agent_i.get_action(last_timestep.observation, act_key, last_timestep.action_mask)[0]
            action_j = agent_j.get_action(last_timestep.observation, act_key, last_timestep.action_mask)[0]
            
            # Remove batch dimension from last_timestep after agent processing
            last_timestep = jax.tree.map(lambda x: jnp.squeeze(x, axis=0), last_timestep)
            
            action = jnp.where(last_timestep.current_player == 0, action_i, action_j)

            # step env
            env_state, new_timestep = env.step(env_state, action)

            # update carry
            acc_reward += new_timestep.reward[0]
            total_return = jnp.where(new_timestep.done, total_return + acc_reward, total_return)
            num_eps = jnp.where(new_timestep.done, num_eps + 1, num_eps)
            acc_reward = jnp.where(new_timestep.done, 0, acc_reward)

            # return next carry and this transition
            return (new_timestep, env_state, key, acc_reward, total_return, num_eps), None
    
    # the output of rollout will have extra dimesion of (num_steps, ...)
    single_env_rollout_fc = nnx.scan(
        collect_one_env_step, length=num_steps
    )

    # batch for num_envs
    batch_env_rollout_fc = nnx.vmap(single_env_rollout_fc, in_axes=(0, None), out_axes=0)

    # prepare batched keys
    key1, key2 = jax.random.split(key)
    reset_keys = jax.random.split(key1, num_envs)
    init_keys = jax.random.split(key2, num_envs)

    # prepare init env state and timestep
    env_states, timesteps = jax.vmap(env.reset)(reset_keys)
    zeros = jnp.zeros((num_envs, ), dtype=jnp.float32)

    # perform batch rollout, return with shape (num_envs, num_steps, ...)
    (_, _, _, _, total_return, num_eps), _ = batch_env_rollout_fc(
        (timesteps, env_states, init_keys, zeros, zeros, zeros), # carry
        None # empty Ys
    )

    avg_return = jnp.mean(total_return / num_eps)

    return avg_return

@partial(jax.jit, donate_argnums=(1, 2, 3))
def _fictitious_play_step(payoff_matrix: chex.Array, 
                         player_0_beliefs: chex.Array,
                         player_1_beliefs: chex.Array,
                         cumulative_strategy_p0: chex.Array) -> tuple[chex.Array, chex.Array, chex.Array]:
    """
    Single step of fictitious play algorithm (JIT-compiled for performance).
    
    Args:
        payoff_matrix: Player 0's payoff matrix
        player_0_beliefs: Player 0's beliefs about player 1's historical play
        player_1_beliefs: Player 1's beliefs about player 0's historical play
        cumulative_strategy_p0: Cumulative count of player 0's actions
    
    Returns:
        Tuple of (updated_player_0_beliefs, updated_player_1_beliefs, updated_cumulative_strategy_p0)
    """
    # Player 0's belief about player 1's mixed strategy
    p1_mixed_strategy = player_0_beliefs / jnp.sum(player_0_beliefs)
    
    # Player 1's belief about player 0's mixed strategy  
    p0_mixed_strategy = player_1_beliefs / jnp.sum(player_1_beliefs)
    
    # Player 0's best response: compute expected payoff for each action
    expected_payoffs_p0 = payoff_matrix @ p1_mixed_strategy
    
    # Player 1's best response: compute expected payoff for each action
    # Player 1's payoff matrix is the negative transpose of player 0's (zero-sum game)
    expected_payoffs_p1 = -payoff_matrix.T @ p0_mixed_strategy
    
    # Choose best response actions (break ties by choosing first action)
    best_action_p0 = jnp.argmax(expected_payoffs_p0)
    best_action_p1 = jnp.argmax(expected_payoffs_p1)
    
    # Update beliefs based on the actions chosen
    updated_player_0_beliefs = player_0_beliefs.at[best_action_p1].add(1)
    updated_player_1_beliefs = player_1_beliefs.at[best_action_p0].add(1)
    
    # Update cumulative strategy for player 0
    updated_cumulative_strategy_p0 = cumulative_strategy_p0.at[best_action_p0].add(1)
    
    return updated_player_0_beliefs, updated_player_1_beliefs, updated_cumulative_strategy_p0


def compute_meta_strategy(payoff_matrix: chex.Array, num_iteration: int = 1000) -> chex.Array:
    """
    Compute player 0's meta-strategy using fictitious play.
    
    Args:
        payoff_matrix: A 2D array where payoff_matrix[i, j] represents 
                      player 0's payoff when player 0 plays action i and player 1 plays action j
        num_iteration: Number of iterations to run fictitious play
    
    Returns:
        A probability distribution over player 0's actions (meta-strategy)
    """
    chex.assert_rank(payoff_matrix, 2)
    
    num_actions_p0, num_actions_p1 = payoff_matrix.shape
    
    # Initialize belief counters for both players
    # player_0_beliefs[j] = number of times player 1 has played action j
    # player_1_beliefs[i] = number of times player 0 has played action i
    player_0_beliefs = jnp.ones(num_actions_p1)  # Start with uniform prior
    player_1_beliefs = jnp.ones(num_actions_p0)  # Start with uniform prior
    
    # Track the cumulative strategy for player 0
    cumulative_strategy_p0 = jnp.zeros(num_actions_p0)
    
    for iteration in range(num_iteration):
        player_0_beliefs, player_1_beliefs, cumulative_strategy_p0 = _fictitious_play_step(
            payoff_matrix, player_0_beliefs, player_1_beliefs, cumulative_strategy_p0
        )
    
    # Return the empirical frequency as the meta-strategy
    meta_strategy = cumulative_strategy_p0 / num_iteration
    
    return meta_strategy
