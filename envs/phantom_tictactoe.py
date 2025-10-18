"""Phantom Tic-Tac-Toe environment implementation using JAX.

reference open_spiel: https://github.com/google-deepmind/open_spiel/tree/master
This module provides a JAX-based implementation of Phantom Tic-Tac-Toe, an imperfect
information variant where players cannot see their opponent's moves. Players must
deduce opponent positions by discovering which moves are blocked.

Two versions are supported:
- Classic: When a move is blocked, player can try again on the same turn
- Abrupt: When a move is blocked, turn immediately ends

Example:
    >>> env = PhantomTicTacToe(is_abrupt=False)  # Classic version
    >>> key = jax.random.key(42)
    >>> state, timestep = env.reset(key)
    >>> state, timestep = env.step(state, jnp.int32(4))  # center square
"""

from typing import Tuple
from functools import partial
import jax.numpy as jnp
import jax
import chex

import envs.mytypes as env_types
from envs.myspaces import Discrete, Box

@chex.dataclass
class EnvState:
    """Environment state for phantom tic-tac-toe game.
    
    Attributes:
        key: PRNG key for random number generation
        current_player: ID of current player (0 or 1)
        done: Whether the game has ended
        step_cnt: Number of steps taken in the game
        true_board: 3x3 actual game board (-1=empty, 0=player0, 1=player1)
        player_knowledge: What each player knows about board positions (2, 3, 3)
                         -1=unknown, 0=own piece, 1=discovered opponent piece
        winner: ID of winner (-1=no winner, 0=player0, 1=player1, 2=draw)
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


class PhantomTicTacToe(env_types.BaseEnv):
    """Phantom Tic-Tac-Toe environment for two players with imperfect information.
    
    In this variant, players cannot see their opponent's moves directly. They must
    attempt to place pieces and discover opponent positions through blocked moves.
    
    Game Rules:
    -----------
    - Players take turns attempting to place their marks on a 3x3 grid
    - Players can only see their own successful placements
    - When attempting to place on an occupied square:
      * Classic version: Move is rejected, player can try again immediately
      * Abrupt version: Turn ends immediately, opponent gets to move
    - Players learn opponent positions by attempting blocked moves
    - Win condition: First to get three in a row (horizontally, vertically, diagonally)
    - Draw condition: All squares filled without a winner
    
    Observation Space:
    -----------------
    Each player observes a 3x3 grid from their perspective:
    - -1: Unknown square (could be empty or contain opponent piece)  
    - 0: Own piece (successfully placed)
    - 1: Discovered opponent piece (attempted placement was blocked)
    """
    
    def __init__(self, is_abrupt: bool = False):
        """Initialize Phantom Tic-Tac-Toe environment.
        
        Args:
            is_abrupt: If True, uses abrupt rules where blocked moves end turn.
                      If False, uses classic rules where player can try again.
        """
        self.is_abrupt = is_abrupt
    
    @property
    def env_name(self) -> str:
        """Environment name identifier."""
        return f"phantom_tic_tac_toe_{'abrupt' if self.is_abrupt else 'classic'}"

    @property
    def num_agents(self) -> int:
        """Number of agents in the environment."""
        return 2

    @property
    def action_space(self) -> Discrete:
        """Action space: 9 discrete actions for 3x3 board positions."""
        return Discrete(num_categories=9)

    @property
    def observation_space(self) -> Box:
        """Observation space: 3x3 board from player's knowledge perspective."""
        return Box(low=-1, high=1, shape=(3, 3), dtype=jnp.int32)

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
            true_board=-jnp.ones((3, 3), dtype=jnp.int32),  # All empty
            player_knowledge=-jnp.ones((2, 3, 3), dtype=jnp.int32),  # All unknown
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
        row, col = action // 3, action % 3

        # Check if action is valid (in bounds)
        is_valid_position = (action >= 0) & (action < 9)

        # Apply action using conditional logic
        updated_state = jax.lax.cond(
            ~state.done,  # Game is still ongoing
            lambda s: jax.lax.cond(
                is_valid_position,  # Position is valid
                partial(self._process_move, row, col),
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
    
    def _process_move(self, row: chex.Numeric, col: chex.Numeric, state: EnvState) -> EnvState:
        """Process a move attempt at the given position.
        
        Args:
            row: Row index (0-2)
            col: Column index (0-2)
            state: Current environment state
            
        Returns:
            Updated state after processing the move
        """
        current_player = state.current_player
        position_is_empty = state.true_board[row, col] == -1
        
        return jax.lax.cond(
            position_is_empty,
            partial(self._successful_placement, row, col),
            partial(self._blocked_placement, row, col),
            state
        )
    
    def _successful_placement(self, row: chex.Numeric, col: chex.Numeric, state: EnvState) -> EnvState:
        """Handle successful piece placement.
        
        Args:
            row: Row index (0-2)
            col: Column index (0-2)
            state: Current environment state
            
        Returns:
            Updated state after successful placement
        """
        current_player = state.current_player
        
        # Update true board with player's piece
        updated_true_board = state.true_board.at[row, col].set(current_player)
        
        # Update both players' knowledge
        # Current player knows they placed their piece here
        updated_knowledge = state.player_knowledge.at[current_player, row, col].set(0)
        # Opponent learns nothing new from this move (they don't see it)
        
        # Check win condition
        has_won = self._check_winner(updated_true_board, current_player)
        
        # Check for draw (board full)
        is_draw = jnp.all(updated_true_board != -1)
        
        return state.replace(
            true_board=updated_true_board,
            player_knowledge=updated_knowledge,
            winner=jnp.where(has_won, current_player, 
                           jnp.where(is_draw, jnp.int32(2), jnp.int32(-1))),  # 2 = draw
            done=has_won | is_draw,
            step_cnt=state.step_cnt + 1,
            current_player=1 - current_player  # Switch players after successful move
        )
    
    def _blocked_placement(self, row: chex.Numeric, col: chex.Numeric, state: EnvState) -> EnvState:
        """Handle blocked piece placement (position occupied by opponent).
        
        Args:
            row: Row index (0-2)
            col: Column index (0-2)
            state: Current environment state
            
        Returns:
            Updated state after blocked placement
        """
        current_player = state.current_player
        
        # Current player discovers opponent piece at this position
        updated_knowledge = state.player_knowledge.at[current_player, row, col].set(1)
        
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
            3x3 board showing what current player knows
        """
        return state.player_knowledge[state.current_player]

    def _get_action_mask(self, state: EnvState) -> chex.Array:
        """Return boolean mask of valid actions from current player's perspective.
        
        In phantom tic-tac-toe, players can attempt to place on any square they
        haven't confirmed is occupied by the opponent.
        
        Args:
            state: Current environment state
            
        Returns:
            Boolean array of length 9 indicating valid actions
        """
        current_knowledge = state.player_knowledge[state.current_player]
        
        # Player can attempt to place anywhere they haven't discovered an opponent piece
        # They can place on: unknown squares (-1) but NOT on discovered opponent pieces (1)
        # They also cannot place on their own pieces (0)
        can_attempt = (current_knowledge == -1)
        
        return can_attempt.flatten()

    def _check_winner(self, board: chex.Array, player: chex.Numeric) -> chex.Numeric:
        """Check if the player has won on the given board.
        
        Args:
            board: 3x3 game board
            player: Player ID to check for win
            
        Returns:
            Boolean indicating if player has won
        """
        # Check horizontal wins
        horizontal_wins = jnp.any(jnp.all(board == player, axis=1))
        
        # Check vertical wins
        vertical_wins = jnp.any(jnp.all(board == player, axis=0))
        
        # Check diagonal wins
        main_diagonal = jnp.all(jnp.diag(board) == player)
        anti_diagonal = jnp.all(jnp.diag(jnp.fliplr(board)) == player)
        diagonal_wins = main_diagonal | anti_diagonal
        
        return horizontal_wins | vertical_wins | diagonal_wins

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
                state.winner == 2,  # Draw
                jnp.zeros(2, dtype=jnp.float32),
                jnp.where(
                    state.winner == 0,  # Player 0 wins
                    jnp.array([1.0, -1.0], dtype=jnp.float32),
                    jnp.array([-1.0, 1.0], dtype=jnp.float32)  # Player 1 wins
                )
            )
        )


if __name__ == "__main__":
    # Test both versions
    print("=== Testing Classic Version ===")
    env_classic = PhantomTicTacToe(is_abrupt=False)
    key = jax.random.key(42)
    state, timestep = env_classic.reset(key)
    
    print(f"Initial player: {state.current_player}")
    print(f"Player {state.current_player} knowledge:")
    print(timestep.observation)
    print(f"Action mask: {timestep.action_mask}")
    
    # Try a successful move
    state, timestep = env_classic.step(state, jnp.int32(4))  # Center
    print(f"\nAfter placing in center:")
    print(f"Current player: {state.current_player}")
    print(f"Player {state.current_player} knowledge:")
    print(timestep.observation)
    
    # Try a blocked move (opponent tries center)
    state, timestep = env_classic.step(state, jnp.int32(4))  # Same position
    print(f"\nAfter blocked move (center):")
    print(f"Current player: {state.current_player}")  # Should be same in classic
    print(f"Player {state.current_player} knowledge:")
    print(timestep.observation)  # Should show discovered opponent piece
    
    print("\n=== Testing Abrupt Version ===")
    env_abrupt = PhantomTicTacToe(is_abrupt=True)
    key = jax.random.key(42)
    state, timestep = env_abrupt.reset(key)
    
    print(f"Initial player: {state.current_player}")
    
    # Place in center
    state, timestep = env_abrupt.step(state, jnp.int32(4))
    print(f"After successful move, current player: {state.current_player}")
    
    # Try blocked move - should switch players immediately in abrupt version  
    state, timestep = env_abrupt.step(state, jnp.int32(4))
    print(f"After blocked move, current player: {state.current_player}")  # Should switch