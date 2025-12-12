"""
Control Variate Variance Reduction for NashPG

This module implements the CV gradient correction:
  g_corrected = g_pi - cv_coefficient * (IS_weighted_g_rho - g_bar_rho)

Where:
- g_pi: Standard policy gradient from current policy
- g_rho: Policy gradient through reference policy (rho)
- IS_weighted_g_rho: g_rho weighted by importance sampling ratio rho(a|s)/pi(a|s)
- g_bar_rho: Baseline gradient computed when pi = rho (stored from snapshot phase)
"""

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
import chex

import train.mytypes as train_types
from agents import BaseAgent


def _masked_mean(x: chex.Array, mask: chex.Array) -> chex.Numeric:
    """Compute mean only over valid (masked) samples"""
    masked_x = x * mask
    return jnp.sum(masked_x) / jnp.maximum(jnp.sum(mask), 1.0)


def _masked_std(x: chex.Array, mask: chex.Array) -> chex.Numeric:
    """Compute standard deviation only over valid (masked) samples"""
    mean_val = _masked_mean(x, mask)
    variance = jnp.sum(mask * jnp.square(x - mean_val)) / jnp.maximum(jnp.sum(mask), 1.0)
    return jnp.sqrt(variance)


def _normalize_advantages(advantage: chex.Array, valid_mask: chex.Array) -> chex.Array:
    """Normalize advantages using masked statistics"""
    adv_mean = _masked_mean(advantage, valid_mask)
    adv_std = _masked_std(advantage, valid_mask)
    return (advantage - adv_mean) / (adv_std + 1e-8)


def compute_baseline_gradient(
    rho_graphdef: Any,
    rho_state: Any,
    dataset: train_types.Dataset,
    ent_coef: float,
) -> Any:
    """
    Compute baseline gradient g_bar_rho at start of outer iteration when pi = rho.
    IS ratio = 1.0, so this is unbiased with no IS variance.

    CONSISTENCY: Use differentiable rho copy (via nnx.merge) even though pi = rho
    at snapshot time. This keeps gradient computation consistent with
    compute_is_weighted_rho_gradient.

    Args:
        rho_graphdef: GraphDef for reconstructing rho
        rho_state: State for reconstructing rho
        dataset: Trajectories collected from agent (= rho at snapshot time)
        ent_coef: Entropy coefficient

    Returns:
        PyTree of gradients matching agent parameter structure
    """
    # Reconstruct differentiable rho for consistency with IS-weighted computation
    rho_agent = nnx.merge(rho_graphdef, rho_state)

    # Pre-normalize advantages (outside the grad computation for consistency)
    normalized_adv = _normalize_advantages(dataset.advantage, dataset.valid_mask)

    def policy_gradient_loss(rho_agent: BaseAgent) -> chex.Numeric:
        dists = rho_agent.get_action_distribution(dataset.observation, dataset.action_mask)
        log_prob = dists.log_prob(dataset.action)

        pg_loss = -_masked_mean(log_prob * normalized_adv, dataset.valid_mask)
        entropy_loss = -_masked_mean(dists.entropy(), dataset.valid_mask)

        return pg_loss + ent_coef * entropy_loss

    rho_grad = nnx.grad(policy_gradient_loss)(rho_agent)
    _, rho_grad_state = nnx.split(rho_grad)
    return rho_grad_state


def compute_is_weighted_rho_gradient(
    rho_graphdef: Any,
    rho_state: Any,
    pi_agent: BaseAgent,
    batch: train_types.Dataset,
    ent_coef: float,
    is_clip: float,
) -> Tuple[Any, chex.Array]:
    """
    Compute IS-weighted gradient through rho using pi's trajectories.

    Steps:
    1. Reconstruct differentiable rho via nnx.merge(graphdef, state)
    2. Compute IS ratio: rho(a|s) / pi(a|s)  [CORRECT: rho/pi for off-policy correction]
    3. Clip IS ratio for stability
    4. Compute grad_rho [IS_ratio * log rho(a|s) * A] with stop_gradient on IS ratio

    NOTE ON BIAS: Per-step IS corrects the action distribution mismatch but NOT
    the state distribution mismatch. This introduces a small bias that is acceptable
    when pi ~= rho (enforced by KL regularization with mag_coef). The bias grows as
    pi diverges from rho during the inner loop, but this is bounded by the IS clipping.

    Args:
        rho_graphdef: GraphDef for reconstructing rho
        rho_state: State for reconstructing rho
        pi_agent: Current policy (for computing IS ratio)
        batch: Current minibatch of data
        ent_coef: Entropy coefficient
        is_clip: Maximum IS ratio (for stability)

    Returns:
        Tuple of (IS-weighted gradient PyTree, mean IS ratio for logging)
    """
    # Reconstruct differentiable rho
    rho_agent = nnx.merge(rho_graphdef, rho_state)

    # Compute IS ratios: rho(a|s) / pi(a|s) for correcting pi->rho distribution shift
    pi_dist = pi_agent.get_action_distribution(batch.observation, batch.action_mask)
    rho_dist = rho_agent.get_action_distribution(batch.observation, batch.action_mask)

    pi_log_prob = pi_dist.log_prob(batch.action)
    rho_log_prob = rho_dist.log_prob(batch.action)

    log_is_ratio = rho_log_prob - pi_log_prob
    is_ratio = jnp.exp(log_is_ratio)

    # Clip IS ratio for stability
    is_ratio_clipped = jnp.clip(is_ratio, 1.0 / is_clip, is_clip)

    # Pre-normalize advantages
    normalized_adv = _normalize_advantages(batch.advantage, batch.valid_mask)

    def rho_policy_gradient_loss(rho_agent: BaseAgent) -> chex.Numeric:
        """
        Compute IS-weighted policy gradient through rho.

        Loss = -E_pi[(rho/pi) * log rho(a|s) * A(s,a)]

        The gradient of this w.r.t rho parameters gives us:
        grad_rho E_pi[(rho/pi) * log rho * A] = E_pi[(rho/pi) * grad_rho log rho * A]

        Note: We use stop_gradient on IS ratio since we want gradient w.r.t. rho only
        """
        rho_dist = rho_agent.get_action_distribution(batch.observation, batch.action_mask)
        rho_log_prob = rho_dist.log_prob(batch.action)

        # IS-weighted policy gradient
        # stop_gradient on IS ratio: we only want gradient through log_prob
        weighted_pg = jax.lax.stop_gradient(is_ratio_clipped) * rho_log_prob * normalized_adv
        pg_loss = -_masked_mean(weighted_pg, batch.valid_mask)

        # Entropy (also IS-weighted for consistency)
        entropy = rho_dist.entropy()
        weighted_entropy = jax.lax.stop_gradient(is_ratio_clipped) * entropy
        entropy_loss = -_masked_mean(weighted_entropy, batch.valid_mask)

        return pg_loss + ent_coef * entropy_loss

    # Compute gradient w.r.t rho parameters
    rho_grad = nnx.grad(rho_policy_gradient_loss)(rho_agent)

    # Extract just the gradient pytree (without module structure)
    _, rho_grad_state = nnx.split(rho_grad)

    # Compute mean IS ratio for logging
    mean_is_ratio = _masked_mean(is_ratio_clipped, batch.valid_mask)

    return rho_grad_state, mean_is_ratio


def apply_cv_correction(
    pi_grad: Any,
    is_weighted_rho_grad: Any,
    baseline_grad: Any,
    cv_coefficient: float,
) -> Tuple[Any, Any]:
    """
    Apply control variate correction to policy gradient.

    g_corrected = g_pi - cv_coefficient * (IS_weighted_g_rho - g_bar_rho)

    Args:
        pi_grad: Gradient from current policy pi
        is_weighted_rho_grad: IS-weighted gradient from rho on current data
        baseline_grad: Baseline gradient g_bar_rho from snapshot phase
        cv_coefficient: Scaling factor for correction (1.0 = full correction)

    Returns:
        Tuple of (corrected gradient PyTree, correction term for logging)
    """
    def compute_correction(g_rho_is, g_rho_bar):
        return cv_coefficient * (g_rho_is - g_rho_bar)

    def correct_leaf(g_pi, g_rho_is, g_rho_bar):
        correction = cv_coefficient * (g_rho_is - g_rho_bar)
        return g_pi - correction

    correction = jax.tree.map(compute_correction, is_weighted_rho_grad, baseline_grad)
    corrected_grad = jax.tree.map(correct_leaf, pi_grad, is_weighted_rho_grad, baseline_grad)

    return corrected_grad, correction


def compute_gradient_norm_metrics(
    pi_grad: Any,
    cv_grad: Any,
    correction: Any,
) -> Dict[str, chex.Array]:
    """
    Compute gradient norm metrics for paper reporting.

    Args:
        pi_grad: Gradient from current policy pi (before CV)
        cv_grad: CV-corrected gradient
        correction: The CV correction term c * (g_rho_IS - g_bar_rho)

    Returns:
        Dict with:
        - cv_grad_norm_before: ||g_pi|| - L2 norm of uncorrected gradient
        - cv_grad_norm_after: ||g_CV|| - L2 norm of CV-corrected gradient
        - cv_correction_norm: ||c * (g_rho_IS - g_bar_rho)|| - L2 norm of correction term
    """
    def compute_norm(grad_pytree: Any) -> chex.Array:
        leaves = jax.tree.leaves(grad_pytree)
        squared_norms = [jnp.sum(jnp.square(leaf)) for leaf in leaves]
        return jnp.sqrt(jnp.sum(jnp.array(squared_norms)))

    return {
        'cv_grad_norm_before': compute_norm(pi_grad),
        'cv_grad_norm_after': compute_norm(cv_grad),
        'cv_correction_norm': compute_norm(correction),
    }
