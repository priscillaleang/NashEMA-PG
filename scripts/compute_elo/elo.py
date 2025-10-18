"""ELO rating system functions."""

from typing import List, Tuple
import numpy as np

from scripts.compute_elo.config import ELO_K_FACTOR
from scripts.compute_elo.elo_agents import Player


def expected_score(elo_a: float, elo_b: float) -> float:
    """Calculate expected score for player A against player B using ELO formula"""
    return 1.0 / (1.0 + 10**((elo_b - elo_a) / 400))


def update_elo(player_a: Player, player_b: Player, score_a: float):
    """Update ELO ratings after a game. score_a: 1.0 = A wins, 0.0 = B wins, 0.5 = draw"""
    expected_a = expected_score(player_a.elo, player_b.elo)
    expected_b = 1.0 - expected_a
    
    # Update ratings
    player_a.elo = player_a.elo + ELO_K_FACTOR * (score_a - expected_a)
    player_b.elo = player_b.elo + ELO_K_FACTOR * ((1.0 - score_a) - expected_b)


def swiss_tournament_pairing(players: List[Player]) -> List[Tuple[Player, Player]]:
    """Generate pairings for Swiss tournament. Returns list of (player_a, player_b) tuples"""
    # Sort players by current ELO rating (highest first)
    sorted_players = sorted(players, key=lambda p: p.elo, reverse=True)
    
    pairings = []
    used_players = set()
    
    for i, player_a in enumerate(sorted_players):
        if player_a in used_players:
            continue
            
        # Find best opponent from remaining players with similar rating
        best_opponent = None
        for j in range(i + 1, len(sorted_players)):
            player_b = sorted_players[j]
            if player_b not in used_players:
                best_opponent = player_b
                break
        
        if best_opponent is not None:
            pairings.append((player_a, best_opponent))
            used_players.add(player_a)
            used_players.add(best_opponent)
    
    return pairings


def compute_elo_by_step(players: List[Player]):
    """Compute average ELO for each algorithm family by training step"""
    # Group players by family and step
    family_step_elos = {}
    
    for player in players:
        key = (player.family, player.step)
        if key not in family_step_elos:
            family_step_elos[key] = []
        family_step_elos[key].append(player.elo)
    
    # Calculate average ELO for each family/step combination
    results = {}
    for (family, step), elos in family_step_elos.items():
        avg_elo = np.mean(elos)
        if family not in results:
            results[family] = []
        results[family].append((step, avg_elo))
    
    # Sort by step for each family
    for family in results:
        results[family].sort(key=lambda x: x[0])
    
    return results