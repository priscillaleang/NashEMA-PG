"""Utility functions and constants for Battleship environment."""

import jax
import jax.numpy as jnp
import chex


# Ship configuration constants
SHIP_SIZES = jnp.array([5, 4, 3, 3, 2])  # [Carrier, Battleship, Cruiser, Submarine, Destroyer]
SHIP_NAMES = ["Carrier", "Battleship", "Cruiser", "Submarine", "Destroyer"]
TOTAL_SHIP_POSITIONS = 17  # Sum of all ship sizes

# Board configuration
BOARD_SIZE = 10
TOTAL_POSITIONS = BOARD_SIZE * BOARD_SIZE

# Stage constants
SETUP_STAGE = 0
PLAY_STAGE = 1

# Board encoding values
EMPTY = 0
SHIP_HEAD = 1
MISS = -1
HIT = 1

# Ship type markers (negative values)
CARRIER_MARKER = -1
BATTLESHIP_MARKER = -2
CRUISER_MARKER = -3
SUBMARINE_MARKER = -4
DESTROYER_MARKER = -5


def count_ships_placed(ship_board: chex.Array) -> chex.Numeric:
    """Count how many ship positions have been placed on the board.
    
    Args:
        ship_board: (10, 10) ship board with negative values for placed ships
        
    Returns:
        Number of ship positions placed
    """
    return jnp.sum(ship_board < 0)


def get_current_ship_being_placed(ship_board: chex.Array) -> chex.Numeric:
    """Get the index of the ship currently being placed. 
    We guarantee that argmax always find match
    
    Args:
        ship_board: (10, 10) ship board
        
    Returns:
        Ship index (0-4) of the ship being placed
    """
    accum_ship_sizes = jnp.array([0, 5, 9, 12, 15], dtype=jnp.int32)
    total_size_on_board = count_ships_placed(ship_board)
    return jnp.argmax(accum_ship_sizes == total_size_on_board)


def is_placing_tail(ship_board: chex.Array) -> chex.Numeric:
    """Check if currently placing a ship's tail.
    
    Args:
        ship_board: (10, 10) ship board
        
    Returns:
        Boolean indicating if placing tail (True) or head (False)
    """
    return jnp.any(ship_board == SHIP_HEAD)


def get_rewards(state_done: chex.Numeric, winner: chex.Numeric) -> chex.Array:
    """Calculate rewards for both players.
    
    Args:
        state_done: Whether the game is finished
        winner: Winner index (-1 if no winner)
        
    Returns:
        Array of rewards for both players
    """
    # No reward during game
    ongoing_reward = jnp.zeros(2, dtype=jnp.float32)
    
    # Win/loss rewards when game ends
    def end_rewards():
        rewards = jnp.zeros(2, dtype=jnp.float32)
        rewards = rewards.at[winner].set(1.0)  # Winner gets +1
        rewards = rewards.at[1 - winner].set(-1.0)  # Loser gets -1
        return rewards
    
    return jax.lax.cond(
        state_done,
        end_rewards,
        lambda: ongoing_reward
    )