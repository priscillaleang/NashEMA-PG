"""
Poker Hand Evaluation Module

Card Encoding:
    - Cards represented as integers 0-51
    - Rank: card % 13 (0=2, 1=3, ..., 11=King, 12=Ace)  
    - Suit: card // 13 (0=Clubs, 1=Diamonds, 2=Hearts, 3=Spades)
    
Hand Strength Scoring:
    - Higher scores indicate stronger hands
    - Scores are comparable across all possible hands
    - Uses lexicographic ordering within hand types
"""

from itertools import combinations
import chex
import jax
import jax.numpy as jnp

def get_suit(card):
    """
    Extract the suit of a card from its integer encoding.
    
    Args:
        card: Integer card representation [0-51]
        
    Returns:
        Suit index: 0=Clubs, 1=Diamonds, 2=Hearts, 3=Spades
    """
    return card // 13

def get_rank(card):
    """
    Extract the rank of a card from its integer encoding.
    
    Args:
        card: Integer card representation [0-51]
        
    Returns:
        Rank index: 0=2, 1=3, ..., 11=King, 12=Ace
    """
    return card % 13

def evaluate_five_cards(cards):
    """
    Evaluate a five-card poker hand and return a numerical strength score.
    
    Determines hand type (pair, flush, straight, etc.) and generates a comparable
    score where higher values indicate stronger hands. Uses lexicographic ordering
    within hand types based on relevant card ranks.
    
    Args:
        cards: Array of shape (5,) containing card integers [0-51]
    
    Returns:
        Integer score where higher = stronger hand. Scores are comparable
        across all possible 5-card combinations.
    """
    suits = get_suit(cards)
    ranks = get_rank(cards)
    sorted_ranks = jnp.sort(ranks)
    
    # Check for flush: all cards of the same suit
    is_flush = (suits == suits[0]).all()
    
    # Check for straight
    no_duplicates = jnp.all(sorted_ranks[1:] != sorted_ranks[:-1])
    is_straight = ((sorted_ranks[4] - sorted_ranks[0] == 4) & no_duplicates) | \
                (jnp.all(sorted_ranks == jnp.array([0, 1, 2, 3, 12])))
    straight_high = jax.lax.cond(
        jnp.all(sorted_ranks == jnp.array([0, 1, 2, 3, 12])),
        lambda _: jnp.int32(3),  # 5-high straight (A,2,3,4,5)
        lambda _: sorted_ranks[4],
        operand=None
    )
    
    # Compute rank counts for pairs, three of a kind, etc.
    rank_counts = jnp.bincount(ranks, length=13)
    max_count = jnp.max(rank_counts)
    count_of_counts = jnp.bincount(rank_counts, length=5)
    
    # Initialize score array
    score_array = jnp.array([-1, -1, -1, -1, -1, -1], dtype=jnp.int32)
    
    # Royal Flush
    score_array = jax.lax.cond(
        is_flush & is_straight & (straight_high == 12),
        lambda _: jnp.array([9, 12, 0, 0, 0, 0], dtype=jnp.int32),
        lambda _: score_array,
        operand=None
    )
    
    # Straight Flush
    score_array = jax.lax.cond(
        (score_array[0] == -1) & is_flush & is_straight,
        lambda _: jnp.array([8, straight_high, 0, 0, 0, 0], dtype=jnp.int32),
        lambda _: score_array,
        operand=None
    )
    
    # Four of a Kind
    four_rank = jnp.argmax(rank_counts == 4)
    masked_ranks = jnp.where(ranks != four_rank, ranks, -1)
    kicker_four = jnp.max(masked_ranks)
    score_array = jax.lax.cond(
        (score_array[0] == -1) & (max_count == 4),
        lambda _: jnp.array([7, four_rank, kicker_four, 0, 0, 0], dtype=jnp.int32),
        lambda _: score_array,
        operand=None
    )
    
    # Full House
    three_rank = jnp.argmax(rank_counts == 3)
    pair_rank = jnp.argmax(rank_counts == 2)
    score_array = jax.lax.cond(
        (score_array[0] == -1) & (max_count == 3) & (count_of_counts[2] >= 1),
        lambda _: jnp.array([6, three_rank, pair_rank, 0, 0, 0], dtype=jnp.int32),
        lambda _: score_array,
        operand=None
    )
    
    # Flush
    score_array = jax.lax.cond(
        (score_array[0] == -1) & is_flush,
        lambda _: jnp.array([5, sorted_ranks[4], sorted_ranks[3], sorted_ranks[2],
                             sorted_ranks[1], sorted_ranks[0]], dtype=jnp.int32),
        lambda _: score_array,
        operand=None
    )
    
    # Straight
    score_array = jax.lax.cond(
        (score_array[0] == -1) & is_straight,
        lambda _: jnp.array([4, straight_high, 0, 0, 0, 0], dtype=jnp.int32),
        lambda _: score_array,
        operand=None
    )
    
    # Three of a Kind
    masked_ranks_three = jnp.where(ranks != three_rank, ranks, -1)
    sorted_masked_three = jnp.sort(masked_ranks_three)[::-1]
    kickers_three = sorted_masked_three[:2]
    score_array = jax.lax.cond(
        (score_array[0] == -1) & (max_count == 3),
        lambda _: jnp.array([3, three_rank, kickers_three[0], kickers_three[1], 0, 0], dtype=jnp.int32),
        lambda _: score_array,
        operand=None
    )
    
    # Two Pairs
    pair_ranks = jnp.argsort(rank_counts)[::-1]
    high_pair = jnp.where(rank_counts[pair_ranks[0]] == 2, pair_ranks[0], -1)
    low_pair = jnp.where(rank_counts[pair_ranks[1]] == 2, pair_ranks[1], -1)
    masked_ranks_two_pairs = jnp.where((ranks != high_pair) & (ranks != low_pair), ranks, -1)
    kicker_two_pairs = jnp.max(masked_ranks_two_pairs)
    score_array = jax.lax.cond(
        (score_array[0] == -1) & (count_of_counts[2] == 2),
        lambda _: jnp.array([2, high_pair, low_pair, kicker_two_pairs, 0, 0], dtype=jnp.int32),
        lambda _: score_array,
        operand=None
    )
    
    # One Pair
    pair_rank_one = jnp.argmax(rank_counts == 2)
    masked_ranks_one_pair = jnp.where(ranks != pair_rank_one, ranks, -1)
    sorted_masked_one_pair = jnp.sort(masked_ranks_one_pair)[::-1]
    kickers_one = sorted_masked_one_pair[:3]
    score_array = jax.lax.cond(
        (score_array[0] == -1) & (max_count == 2),
        lambda _: jnp.array([1, pair_rank_one, kickers_one[0], kickers_one[1],
                             kickers_one[2], 0], dtype=jnp.int32),
        lambda _: score_array,
        operand=None
    )
    
    # High Card
    score_array = jax.lax.cond(
        score_array[0] == -1,
        lambda _: jnp.array([0, sorted_ranks[4], sorted_ranks[3], sorted_ranks[2],
                             sorted_ranks[1], sorted_ranks[0]], dtype=jnp.int32),
        lambda _: score_array,
        operand=None
    )
    
    # Convert score array to single integer using vectorized operation
    weights = jnp.array([13 ** 5, 13 ** 4, 13 ** 3, 13 ** 2, 13 ** 1, 13 ** 0])
    score = jnp.dot(score_array, weights)
    return score

def evaluate_hand(hole_cards: chex.Array, community_cards: chex.Array) -> chex.Array:
    """
    Evaluate the best possible 5-card poker hand from 7 available cards.
    
    Combines player's 2 hole cards with 5 community cards to form all possible
    5-card combinations (21 total), evaluates each combination, and returns the
    score of the strongest hand.
    
    Args:
        hole_cards: Array of shape (2,) containing player's private cards [0-51]
        community_cards: Array of shape (5,) containing board cards [0-51]
        
    Returns:
        Integer score of the best possible hand (higher = stronger)
    """
    # Generate all 21 possible 5-card combinations from 7 cards
    comb_indices = jnp.array(list(combinations(range(7), 5)))
    all_cards = jnp.concatenate([hole_cards, community_cards])  # Shape (7,)
    selected_cards = all_cards[comb_indices]  # Shape (21, 5)
    
    # Evaluate all combinations and return the best score
    scores = jax.vmap(evaluate_five_cards)(selected_cards)  # Shape (21,)
    best_score = jnp.max(scores)
    return best_score