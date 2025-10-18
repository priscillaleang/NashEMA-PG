"""Dark Hex 3 environment implementation using JAX.

reference open_spiel: https://github.com/google-deepmind/open_spiel/tree/master
This module provides a JAX-based implementation of Dark Hex 3, an imperfect
information variant of Hex played on a 3x3 hexagonal grid where players cannot 
see their opponent's stones. Players must connect opposite edges of the board.

Two versions are supported:
- Classic: When a move is blocked, player can try again on the same turn
- Abrupt: When a move is blocked, turn immediately ends

Example:
    >>> env = DarkHex3(is_abrupt=False)  # Classic version
    >>> key = jax.random.PRNGKey(42)
    >>> state, timestep = env.reset(key)
    >>> state, timestep = env.step(state, jnp.int32(4))  # center position
"""

from typing import Tuple
from functools import partial
import jax.numpy as jnp
import jax
import chex

import envs.mytypes as env_types
from envs.myspaces import Discrete, Box

# Board topology for 3x3 hexagonal grid
# Positions:  0 1 2
#              3 4 5  
#               6 7 8
NEIGHBORS = jnp.array([
    [1, 3, -1, -1, -1, -1],  # Position 0: neighbors [1, 3]
    [0, 2, 3, 4, -1, -1],    # Position 1: neighbors [0, 2, 3, 4] 
    [1, 4, 5, -1, -1, -1],   # Position 2: neighbors [1, 4, 5]
    [0, 1, 4, 6, -1, -1],    # Position 3: neighbors [0, 1, 4, 6]
    [1, 2, 3, 5, 6, 7],      # Position 4: neighbors [1, 2, 3, 5, 6, 7]
    [2, 4, 7, 8, -1, -1],    # Position 5: neighbors [2, 4, 7, 8]
    [3, 4, 7, -1, -1, -1],   # Position 6: neighbors [3, 4, 7]
    [4, 5, 6, 8, -1, -1],    # Position 7: neighbors [4, 5, 6, 8]
    [5, 7, -1, -1, -1, -1],  # Position 8: neighbors [5, 7]
], dtype=jnp.int32)

# Edge definitions for each player
# Player 0: connects top edge {0, 1, 2} to bottom edge {6, 7, 8}
# Player 1: connects left edge {0, 3, 6} to right edge {2, 5, 8}
PLAYER_START_EDGES = jnp.array([
    [0, 1, 2],  # Player 0 start edge (top)
    [0, 3, 6],  # Player 1 start edge (left)  
], dtype=jnp.int32)

PLAYER_END_EDGES = jnp.array([
    [6, 7, 8],  # Player 0 end edge (bottom)
    [2, 5, 8],  # Player 1 end edge (right)
], dtype=jnp.int32)

# Precomputed winning paths for each player (padded to length 5 with -1)
# Player 0: connects top edge {0, 1, 2} to bottom edge {6, 7, 8}
PLAYER0_WINNING_PATHS = jnp.array([
    [0, 3, 6, -1, -1],      # Path 1: 0 -> 3 -> 6
    [1, 3, 6, -1, -1],      # Path 2: 1 -> 3 -> 6
    [1, 4, 6, -1, -1],      # Path 3: 1 -> 4 -> 6
    [1, 4, 7, -1, -1],      # Path 4: 1 -> 4 -> 7
    [2, 4, 6, -1, -1],      # Path 5: 2 -> 4 -> 6
    [2, 4, 7, -1, -1],      # Path 6: 2 -> 4 -> 7
    [2, 5, 7, -1, -1],      # Path 7: 2 -> 5 -> 7
    [2, 5, 8, -1, -1],      # Path 8: 2 -> 5 -> 8
    [0, 3, 4, 7, -1],       # Path 9: 0 -> 3 -> 4 -> 7
    [1, 4, 5, 8, -1],       # Path 10: 1 -> 4 -> 5 -> 8
    [0, 3, 4, 5, 8],        # Path 11: 0 -> 3 -> 4 -> 5 -> 8
], dtype=jnp.int32)

# Player 1: connects left edge {0, 3, 6} to right edge {2, 5, 8}
PLAYER1_WINNING_PATHS = jnp.array([
    [0, 1, 2, -1, -1],      # Path 1: 0 -> 1 -> 2
    [3, 1, 2, -1, -1],      # Path 2: 3 -> 1 -> 2
    [3, 4, 2, -1, -1],      # Path 3: 3 -> 4 -> 2
    [3, 4, 5, -1, -1],      # Path 4: 3 -> 4 -> 5
    [6, 4, 2, -1, -1],      # Path 5: 6 -> 4 -> 2
    [6, 4, 5, -1, -1],      # Path 6: 6 -> 4 -> 5
    [6, 7, 5, -1, -1],      # Path 7: 6 -> 7 -> 5
    [6, 7, 8, -1, -1],      # Path 8: 6 -> 7 -> 8
    [0, 1, 4, 5, -1],       # Path 9: 0 -> 1 -> 4 -> 5
    [3, 4, 7, 8, -1],       # Path 10: 3 -> 4 -> 7 -> 8
    [0, 1, 4, 7, 8],        # Path 11: 0 -> 1 -> 4 -> 7 -> 8
], dtype=jnp.int32)

@chex.dataclass
class EnvState:
    """Environment state for dark hex 3 game.
    
    Attributes:
        key: PRNG key for random number generation
        current_player: ID of current player (0 or 1)
        done: Whether the game has ended
        step_cnt: Number of steps taken in the game
        true_board: 9-element actual game board (-1=empty, 0=player0, 1=player1)
        player_knowledge: What each player knows about board positions (2, 9)
                         -1=unknown, 0=own stone, 1=discovered opponent stone
        winner: ID of winner (-1=no winner, 0=player0, 1=player1)
        is_abrupt_version: Whether using abrupt rules (turn ends on blocked moves)
    """
    key: chex.PRNGKey
    current_player: chex.Numeric
    done: chex.Numeric
    step_cnt: chex.Numeric
    true_board: chex.Array
    player_knowledge: chex.Array
    winner: chex.Numeric
    is_abrupt_version: chex.Numeric


class DarkHex3(env_types.BaseEnv):
    """Dark Hex 3 environment for two players with imperfect information.
    
    A variant of Hex played on a 3×3 hexagonal grid where players cannot see
    opponent stones. Each player must connect their assigned opposite edges.
    
    Game Rules:
    -----------
    - 3x3 hexagonal grid with positions 0-8
    - Player 0: connects top edge {0,1,2} to bottom edge {6,7,8}
    - Player 1: connects left edge {0,3,6} to right edge {2,5,8}  
    - Players alternate placing stones but cannot see opponent stones
    - When attempting to place on occupied square:
      * Classic version: Move rejected, player tries again immediately
      * Abrupt version: Turn ends immediately, opponent gets to move
    - Players learn opponent positions by attempting blocked moves
    - Win condition: First to connect their opposite edges wins
    - No draws possible due to Hex topology
    
    Board Layout:
    -------------
    Positions:  0 1 2
                 3 4 5  
                  6 7 8
                  
    Observation Space:
    -----------------
    Each player observes a 9-element array from their perspective:
    - -1: Unknown position (could be empty or contain opponent stone)  
    - 0: Own stone (successfully placed)
    - 1: Discovered opponent stone (attempted placement was blocked)
    """
    
    def __init__(self, is_abrupt: bool = False):
        """Initialize Dark Hex 3 environment.
        
        Args:
            is_abrupt: If True, uses abrupt rules where blocked moves end turn.
                      If False, uses classic rules where player can try again.
        """
        self.is_abrupt = is_abrupt
    
    @property
    def env_name(self) -> str:
        """Environment name identifier."""
        return f"dark_hex3_{'abrupt' if self.is_abrupt else 'classic'}"

    @property
    def num_agents(self) -> int:
        """Number of agents in the environment."""
        return 2

    @property
    def action_space(self) -> Discrete:
        """Action space: 9 discrete actions for 3x3 hex grid positions."""
        return Discrete(num_categories=9)

    @property
    def observation_space(self) -> Box:
        """Observation space: 9-element board from player's knowledge perspective."""
        return Box(low=-1, high=1, shape=(9,), dtype=jnp.int32)

    @partial(jax.jit, static_argnums=0)
    def reset(self, key: chex.PRNGKey) -> Tuple[EnvState, env_types.TimeStep]:
        """Reset the environment to initial state.
        
        Args:
            key: PRNG key for random initialization
            
        Returns:
            Tuple of (initial_state, initial_timestep)
        """
        key, player_key = jax.random.split(key)
        starting_player = jax.random.bernoulli(player_key).astype(jnp.int32)

        initial_state = EnvState(
            key=key,
            current_player=starting_player,
            done=jnp.bool(False),
            step_cnt=jnp.int32(0),
            true_board=-jnp.ones((9,), dtype=jnp.int32),  # All empty
            player_knowledge=-jnp.ones((2, 9), dtype=jnp.int32),  # All unknown
            winner=jnp.int32(-1),
            is_abrupt_version=jnp.bool(self.is_abrupt)
        )

        initial_timestep = env_types.TimeStep(
            reward=jnp.zeros((2,), dtype=jnp.float32),
            done=initial_state.done,
            observation=self._get_observation(initial_state),
            action_mask=self._get_action_mask(initial_state),
            current_player=initial_state.current_player,
            info={"step_cnt": initial_state.step_cnt}
        )

        return initial_state, initial_timestep
    
    @partial(jax.jit, static_argnums=0)
    def step(self, state: EnvState, action: env_types.Action) -> Tuple[EnvState, env_types.TimeStep]:
        """Execute one step of the environment.
        
        Args:
            state: Current environment state
            action: Action to take (0-8, representing board positions)
            
        Returns:
            Tuple of (new_state, timestep)
        """
        chex.assert_shape(action.shape, ())

        # Check if action is valid (in bounds)
        is_valid_position = (action >= 0) & (action < 9)

        # Apply action using conditional logic
        updated_state = jax.lax.cond(
            ~state.done,  # Game is still ongoing
            lambda s: jax.lax.cond(
                is_valid_position,  # Position is valid
                partial(self._process_move, action),
                self._handle_invalid_position,
                s
            ),
            lambda s: s,  # Don't modify state when game is done
            state
        )

        # Calculate rewards based on game outcome
        rewards = self._calculate_rewards(updated_state)

        final_timestep = env_types.TimeStep(
            reward=rewards,
            done=updated_state.done,
            observation=self._get_observation(updated_state),
            action_mask=self._get_action_mask(updated_state),
            current_player=updated_state.current_player,
            info={"step_cnt": updated_state.step_cnt}
        )
        
        return updated_state, final_timestep
    
    def _process_move(self, position: chex.Numeric, state: EnvState) -> EnvState:
        """Process a move attempt at the given position.
        
        Args:
            position: Position index (0-8)
            state: Current environment state
            
        Returns:
            Updated state after processing the move
        """
        current_player = state.current_player
        position_is_empty = state.true_board[position] == -1
        
        return jax.lax.cond(
            position_is_empty,
            partial(self._successful_placement, position),
            partial(self._blocked_placement, position),
            state
        )
    
    def _successful_placement(self, position: chex.Numeric, state: EnvState) -> EnvState:
        """Handle successful stone placement.
        
        Args:
            position: Position index (0-8)
            state: Current environment state
            
        Returns:
            Updated state after successful placement
        """
        current_player = state.current_player
        
        # Update true board with player's stone
        updated_true_board = state.true_board.at[position].set(current_player)
        
        # Update both players' knowledge
        # Current player knows they placed their stone here
        updated_knowledge = state.player_knowledge.at[current_player, position].set(0)
        
        # Check win condition
        has_won = self._check_winner(updated_true_board, current_player)
        
        return state.replace(
            true_board=updated_true_board,
            player_knowledge=updated_knowledge,
            winner=jnp.where(has_won, current_player, jnp.int32(-1)),
            done=has_won,
            step_cnt=state.step_cnt + 1,
            current_player=1 - current_player  # Switch players after successful move
        )
    
    def _blocked_placement(self, position: chex.Numeric, state: EnvState) -> EnvState:
        """Handle blocked stone placement (position occupied by opponent).
        
        Args:
            position: Position index (0-8)
            state: Current environment state
            
        Returns:
            Updated state after blocked placement
        """
        current_player = state.current_player
        
        # Current player discovers opponent stone at this position
        updated_knowledge = state.player_knowledge.at[current_player, position].set(1)
        
        # Determine next player based on version
        next_player = jax.lax.cond(
            state.is_abrupt_version,
            lambda: 1 - current_player,  # Abrupt: switch to opponent
            lambda: current_player       # Classic: same player continues
        )
        
        return state.replace(
            player_knowledge=updated_knowledge,
            step_cnt=state.step_cnt + 1,
            current_player=next_player
        )
    
    def _handle_invalid_position(self, state: EnvState) -> EnvState:
        """Handle invalid position (out of bounds).
        
        Args:
            state: Current environment state
            
        Returns:
            Updated state with opponent as winner due to invalid move
        """
        return state.replace(
            winner=1 - state.current_player,
            done=jnp.bool(True),
            step_cnt=state.step_cnt + 1,
            current_player=1 - state.current_player
        )

    def _get_observation(self, state: EnvState) -> chex.Array:
        """Get observation from current player's perspective.
        
        Args:
            state: Current environment state
            
        Returns:
            9-element array showing what current player knows
        """
        return state.player_knowledge[state.current_player]

    def _get_action_mask(self, state: EnvState) -> chex.Array:
        """Return boolean mask of valid actions from current player's perspective.
        
        In Dark Hex 3, players can attempt to place on any position they
        haven't confirmed is occupied by the opponent.
        
        Args:
            state: Current environment state
            
        Returns:
            Boolean array of length 9 indicating valid actions
        """
        current_knowledge = state.player_knowledge[state.current_player]
        
        # Player can attempt to place anywhere they haven't discovered an opponent stone
        # They can place on: unknown squares (-1) but NOT on discovered opponent stones (1)
        # They also cannot place on their own stones (0)
        can_attempt = (current_knowledge == -1)
        
        return can_attempt

    def _check_winner(self, board: chex.Array, player: chex.Numeric) -> chex.Numeric:
        """Check if the player has won using precomputed winning paths.
        
        This optimized version replaces the BFS implementation with precomputed
        minimal winning paths, making it more parallelizable and JAX-friendly.
        
        Args:
            board: 9-element game board
            player: Player ID to check for win
            
        Returns:
            Boolean indicating if player has won
        """
        # Select winning paths for the current player
        player_paths = jax.lax.cond(
            player == 0,
            lambda: PLAYER0_WINNING_PATHS,
            lambda: PLAYER1_WINNING_PATHS
        )
        
        def check_path(path):
            """Check if all positions in a winning path belong to the player."""
            # Create mask for valid positions (not -1)
            valid_positions = path >= 0
            
            # Check if all valid positions in the path belong to the player
            path_satisfied = jnp.all(
                jnp.where(valid_positions, board[path] == player, True)
            )
            
            return path_satisfied
        
        # Check if any winning path is satisfied
        return jnp.any(jax.vmap(check_path)(player_paths))

    def _calculate_rewards(self, state: EnvState) -> chex.Array:
        """Calculate rewards based on game outcome.
        
        Args:
            state: Current environment state
            
        Returns:
            Array of rewards for both players
        """
        return jnp.where(
            state.winner == -1,  # No winner yet
            jnp.zeros(2, dtype=jnp.float32),
            jnp.where(
                state.winner == 0,  # Player 0 wins
                jnp.array([1.0, -1.0], dtype=jnp.float32),
                jnp.array([-1.0, 1.0], dtype=jnp.float32)  # Player 1 wins
            )
        )


if __name__ == "__main__":
    # Test both versions
    print("=== Testing Classic Version ===")
    env_classic = DarkHex3(is_abrupt=False)
    key = jax.random.PRNGKey(42)
    state, timestep = env_classic.reset(key)
    
    print(f"Initial player: {state.current_player}")
    print(f"Player {state.current_player} knowledge: {timestep.observation}")
    print(f"Action mask: {timestep.action_mask}")
    
    # Try a successful move
    state, timestep = env_classic.step(state, jnp.int32(4))  # Center
    print(f"\nAfter placing in center:")
    print(f"Current player: {state.current_player}")
    print(f"Player {state.current_player} knowledge: {timestep.observation}")
    
    # Try a blocked move (opponent tries center)
    state, timestep = env_classic.step(state, jnp.int32(4))  # Same position
    print(f"\nAfter blocked move (center):")
    print(f"Current player: {state.current_player}")  # Should be same in classic
    print(f"Player {state.current_player} knowledge: {timestep.observation}")
    
    print("\n=== Testing Abrupt Version ===")
    env_abrupt = DarkHex3(is_abrupt=True)
    key = jax.random.PRNGKey(42)
    state, timestep = env_abrupt.reset(key)
    
    print(f"Initial player: {state.current_player}")
    
    # Place in center
    state, timestep = env_abrupt.step(state, jnp.int32(4))
    print(f"After successful move, current player: {state.current_player}")
    
    # Try blocked move - should switch players immediately in abrupt version  
    state, timestep = env_abrupt.step(state, jnp.int32(4))
    print(f"After blocked move, current player: {state.current_player}")  # Should switch