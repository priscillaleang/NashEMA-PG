"""Tournament logic for ELO computation."""

from typing import List, Tuple
from tqdm import tqdm
from functools import partial

from scripts.compute_elo.elo_agents import Player
from scripts.compute_elo.elo import swiss_tournament_pairing
from scripts.compute_elo.elo import update_elo
from scripts.compute_elo.config import NUM_TOURNAMENT_ROUNDS, GAMES_PER_PAIRING, STEP_PER_GAME
from scripts.compute_elo.elo_agents import MixtureAgent

from flax import nnx
import jax
import jax.numpy as jnp
import chex


def check_players_same_graphdef(players: List[Player]):
    """Check if all players have the same graphdef.
    
    Returns:
        (bool, graphdef): (True if all same, the common graphdef object)
    """
    if len(players) == 0:
        return True, None
    
    # Get reference graphdef from first player
    first_player_agent = players[0].agent
    if isinstance(first_player_agent, MixtureAgent):
        ref_graphdef, _ = nnx.split(first_player_agent.agents[0])
    else:  # Normal agent
        ref_graphdef, _ = nnx.split(first_player_agent)
    
    # Check all players
    for i, player in enumerate(players):
        agent = player.agent
        if isinstance(agent, MixtureAgent):
            # Check all constituent agents have same graphdef
            for j, constituent_agent in enumerate(agent.agents):
                const_graphdef, _ = nnx.split(constituent_agent)
                if ref_graphdef != const_graphdef:
                    return False, ref_graphdef
        else:  # Normal agent
            current_graphdef, _ = nnx.split(agent)
            if ref_graphdef != current_graphdef:
                return False, ref_graphdef
    
    return True, ref_graphdef


def run_swiss_tournament(players: List[Player], env, rngs, num_rounds: int = NUM_TOURNAMENT_ROUNDS):
    """Run a Swiss tournament with the given players"""

    # Assert all players have the same graphdef
    all_same, common_graphdef = check_players_same_graphdef(players)
    assert all_same, "All players must have the same graphdef"

    print(f"Starting Swiss tournament with {len(players)} players for {num_rounds} rounds")
    
    for round_num in tqdm(range(num_rounds), desc="Tournament rounds"):
        # Generate pairings for this round
        pairings = swiss_tournament_pairing(players)
        
        # Initialize accumulated rewards for this round
        all_rewards = jnp.zeros((len(pairings), 2), jnp.float32)

        # run all game
        for game_idx in tqdm(range(GAMES_PER_PAIRING), desc=f"Round {round_num + 1} games", leave=False):

            # Prepare agent parameters for this game (sample from mixture agents if needed)
            agent_params = []
            for p1, p2 in pairings:
                # Get actual agents (sample from mixture if needed)
                agent1 = p1.agent.sample_agent() if isinstance(p1.agent, MixtureAgent) else p1.agent
                agent2 = p2.agent.sample_agent() if isinstance(p2.agent, MixtureAgent) else p2.agent
                
                # Extract parameters
                _, param1 = nnx.split(agent1)
                _, param2 = nnx.split(agent2)
                
                agent_params.append((param1, param2))

            current_rewards = play_match(common_graphdef, env, agent_params, rngs.default())

            all_rewards += current_rewards

        # Update ELO ratings based on accumulated rewards from all games
        for i, (p1, p2) in enumerate(pairings):
            # Calculate match result based on accumulated rewards across all games
            p1_total_reward = all_rewards[i, 0]
            p2_total_reward = all_rewards[i, 1]
            
            if p1_total_reward > p2_total_reward:
                match_score = 1.0  # Player 1 wins overall
            elif p2_total_reward > p1_total_reward:
                match_score = 0.0  # Player 2 wins overall
            else:
                match_score = 0.5  # Draw overall
            
            # Update ELO ratings
            update_elo(p1, p2, match_score)


@partial(jax.jit, static_argnums=(0, 1))
def play_match(graphdef, env, agent_params: List[Tuple[nnx.State, nnx.State]], key: chex.PRNGKey) -> chex.Array:
    """Play matches for multiple pairs of agents simultaneously.
    
    Args:
        graphdef: Common graphdef for all agents
        env: Environment instance
        agent_params: List of (state1, state2) tuples for each pair
        key: JAX random key
    
    Returns:
        Array of shape (num_pairs, 2) containing accumulated rewards for each pair
    """
    num_pairs = len(agent_params)
    key, *pair_keys = jax.random.split(key, num_pairs + 1)
    
    # Vectorize over pairs
    def play_single_pair(state1, state2, pair_key):
        # Reconstruct agents from graphdef and states
        agent1 = nnx.merge(graphdef, state1)
        agent2 = nnx.merge(graphdef, state2)
        
        # Initialize environment
        env_state, timestep = env.reset(pair_key)
        
        # Initialize cumulative rewards
        total_rewards = jnp.zeros(2)
        
        # Game loop for fixed number of steps
        def step_fn(carry, step_i):
            env_state, timestep, total_rewards = carry
            
            # Get action from current agent based on current_player
            batched_obs = jax.tree.map(lambda x: jnp.expand_dims(x, axis=0), timestep.observation)
            batched_action_mask = jnp.expand_dims(timestep.action_mask, axis=0)
            step_key = jax.random.fold_in(pair_key, step_i)
            
            # Use conditional to select action from correct agent
            batched_action = jax.lax.cond(
                timestep.current_player == 0,
                lambda: agent1.get_action(batched_obs, step_key, batched_action_mask),
                lambda: agent2.get_action(batched_obs, step_key, batched_action_mask)
            )
            action = batched_action[0]  # Extract single action from batch
            
            # Step environment
            new_env_state, new_timestep = env.step(env_state, action)
            
            # Accumulate rewards
            new_total_rewards = total_rewards + new_timestep.reward
            
            return (new_env_state, new_timestep, new_total_rewards), None
        
        # Run game for STEP_PER_GAME steps
        (_, _, final_rewards), _ = jax.lax.scan(
            step_fn, 
            (env_state, timestep, total_rewards), 
            jnp.arange(STEP_PER_GAME)
        )
        
        return final_rewards
    
    # Separate agent parameters into two batched pytrees
    states1 = [params[0] for params in agent_params]
    states2 = [params[1] for params in agent_params]
    
    # Stack the states to create batched pytrees
    batched_states1 = jax.tree.map(lambda *states: jnp.stack(states, axis=0), *states1)
    batched_states2 = jax.tree.map(lambda *states: jnp.stack(states, axis=0), *states2)
    
    # Vectorize over all pairs
    all_rewards = jax.vmap(play_single_pair)(batched_states1, batched_states2, jnp.array(pair_keys))
    
    return all_rewards
