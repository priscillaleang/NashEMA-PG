import jax
import jax.numpy as jnp
from flax import nnx

# --- Multi-Head Self-Attention Module ---
class MultiHeadAttention(nnx.Module):
    """
    A multi-head self-attention module without dropout.
    
    Attributes:
        num_heads: The number of attention heads.
        features: The total number of features (d_model).
        qkv_features: The size of the query, key, and value projections.
        out_features: The size of the output projection.
    """
    def __init__(
        self,
        num_heads: int,
        features: int,
        qkv_features: int,
        out_features: int,
        *,
        rngs: nnx.Rngs,
    ):
        self.num_heads = num_heads
        self.qkv_features = qkv_features
        self.out_features = out_features
        
        # Projections for query, key, and value
        self.query_proj = nnx.Linear(features, qkv_features, rngs=rngs)
        self.key_proj = nnx.Linear(features, qkv_features, rngs=rngs)
        self.value_proj = nnx.Linear(features, qkv_features, rngs=rngs)
        
        # Output projection
        self.out_proj = nnx.Linear(qkv_features, out_features, rngs=rngs)
        
    def __call__(self, inputs: jax.Array):
        """
        Performs the forward pass of the multi-head attention.
        
        Args:
            inputs: The input tensor with shape `(batch_size, seq_len, features)`.
            
        Returns:
            The output tensor with shape `(batch_size, seq_len, out_features)`.
        """
        # Project inputs to query, key, and value
        q = self.query_proj(inputs)  # (batch, seq_len, qkv_features)
        k = self.key_proj(inputs)   # (batch, seq_len, qkv_features)
        v = self.value_proj(inputs)   # (batch, seq_len, qkv_features)

        # Split projections into heads
        q = self._split_heads(q)  # (batch, num_heads, seq_len, head_dim)
        k = self._split_heads(k)  # (batch, num_heads, seq_len, head_dim)
        v = self._split_heads(v)  # (batch, num_heads, seq_len, head_dim)
        
        # Compute attention scores
        # scaled_dot_product_attention: (batch, num_heads, seq_len, head_dim)
        attention_output = self._scaled_dot_product_attention(q, k, v)

        # Recombine heads
        attention_output = self._combine_heads(attention_output)  # (batch, seq_len, qkv_features)
        
        # Apply output projection
        output = self.out_proj(attention_output)

        return output

    def _split_heads(self, x: jax.Array):
        """Splits the last dimension into (num_heads, head_dim)."""
        batch_size, seq_len, features = x.shape
        return x.reshape(batch_size, seq_len, self.num_heads, features // self.num_heads).transpose(0, 2, 1, 3)

    def _combine_heads(self, x: jax.Array):
        """Combines the heads back into a single feature dimension."""
        batch_size, num_heads, seq_len, head_dim = x.shape
        return x.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, num_heads * head_dim)

    def _scaled_dot_product_attention(self, q: jax.Array, k: jax.Array, v: jax.Array):
        """
        Computes the scaled dot-product attention without dropout.
        
        Args:
            q, k, v: Tensors with shape `(batch, num_heads, seq_len, head_dim)`.
            
        Returns:
            The attention output tensor.
        """
        # Get head dimension
        head_dim = q.shape[-1]
        
        # Compute raw attention scores
        # (batch, num_heads, seq_len_q, seq_len_k)
        scores = jnp.einsum('bhid,bhjd->bhij', q, k) / jnp.sqrt(head_dim)

        # Apply softmax to get attention weights
        weights = jax.nn.softmax(scores, axis=-1)
        
        # Compute weighted sum of values
        # (batch, num_heads, seq_len_q, head_dim)
        output = jnp.einsum('bhij,bhjd->bhid', weights, v)
        
        return output

# --- Transformer Block Module ---
class TransformerBlock(nnx.Module):
    """
    A typical Transformer Block, revised for RL (no dropout).
    
    Attributes:
        features: The number of features (d_model).
        num_heads: The number of attention heads.
        mlp_dim: The dimension of the hidden layer in the feed-forward network.
    """
    def __init__(
        self,
        features: int,
        num_heads: int,
        mlp_dim: int,
        *,
        rngs: nnx.Rngs,
    ):
        self.features = features
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim

        # Multi-head attention layer
        self.attention = MultiHeadAttention(
            num_heads,
            features,
            qkv_features=features,
            out_features=features,
            rngs=rngs
        )

        # Feed-forward network
        self.mlp = nnx.Sequential(
            nnx.Linear(features, mlp_dim, rngs=rngs),
            nnx.relu,
            nnx.Linear(mlp_dim, features, rngs=rngs)
        )
        
        # Layer normalization layers
        self.norm1 = nnx.LayerNorm(features, rngs=rngs)
        self.norm2 = nnx.LayerNorm(features, rngs=rngs)

    def __call__(self, inputs: jax.Array):
        """
        Performs the forward pass of the transformer block.
        
        Args:
            inputs: The input tensor with shape `(batch_size, seq_len, features)`.
            
        Returns:
            The output tensor with shape `(batch_size, seq_len, features)`.
        """
        # Attention sub-layer with residual connection and layer normalization
        attention_output = self.attention(self.norm1(inputs))
        x = inputs + attention_output
        
        # Feed-forward sub-layer with residual connection and layer normalization
        mlp_output = self.mlp(self.norm2(x))
        output = x + mlp_output
        
        return output


def main():
    # Define hyperparameters for the transformer block
    d_model = 256        # Model dimension
    num_heads = 8        # Number of attention heads
    mlp_dim = 1024       # MLP hidden dimension
    batch_size = 4
    seq_len = 10
    
    # Initialize a JAX PRNG key
    key = jax.random.PRNGKey(42)
    
    # Create RNGs for the nnx module
    # The 'dropout' key is no longer needed
    rngs = nnx.Rngs(params=key)

    # Instantiate the Transformer Block
    transformer_block = TransformerBlock(
        features=d_model,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        rngs=rngs,
    )
    
    # Generate some dummy input data
    dummy_input = jnp.ones((batch_size, seq_len, d_model))
    
    print("--- Initializing Transformer Block (RL version) ---")
    print(f"Input shape: {dummy_input.shape}")
    
    # Run the forward pass
    # Note: The function call is now simplified, without the use_running_state argument
    output = transformer_block(dummy_input)
    
    print("\n--- Forward Pass Complete ---")
    print(f"Output shape: {output.shape}")
    
    # Running the forward pass again yields the exact same output due to determinism
    output2 = transformer_block(dummy_input)
    print("\n--- Second Forward Pass Complete ---")
    print(f"Output shape: {output2.shape}")
    assert jnp.array_equal(output, output2)
    print("The outputs from two consecutive runs are identical (model is deterministic).")

if __name__ == "__main__":
    main()

