"""
Configuration class for the GPT-2 model.
"""

class GPT2Config:
    """
    Configuration object for the scaled-down GPT-2 model.
    
    This class stores the hyperparameters for the model architecture and training.
    """
    def __init__(
        self,
        vocab_size: int,
        n_positions: int = 128,
        n_embd: int = 192,
        n_layer: int = 3,
        n_head: int = 3,
        n_inner: int = None,
        activation_function: str = "gelu",
        resid_pdrop: float = 0.1,
        embd_pdrop: float = 0.1,
        attn_pdrop: float = 0.1,
        layer_norm_epsilon: float = 1e-5,
        initializer_range: float = 0.02,
        bos_token_id: int = None,
        eos_token_id: int = None,
    ):
        """
        Initializes the configuration.

        Args:
            vocab_size (int): The size of the vocabulary.
            n_positions (int): The maximum sequence length (context window).
            n_embd (int): The embedding dimension.
            n_layer (int): The number of transformer blocks.
            n_head (int): The number of attention heads.
            n_inner (int, optional): The inner dimension of the MLP. 
                                     Defaults to 4 * n_embd.
            activation_function (str): The activation function to use in the MLP.
            resid_pdrop (float): Dropout probability for residual connections.
            embd_pdrop (float): Dropout probability for embeddings.
            attn_pdrop (float): Dropout probability for attention weights.
            layer_norm_epsilon (float): Epsilon for LayerNorm.
            initializer_range (float): Stddev for weight initialization.
            bos_token_id (int, optional): The ID of the beginning-of-sentence token.
            eos_token_id (int, optional): The ID of the end-of-sentence token.
        """
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = 4 * n_embd if n_inner is None else n_inner
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"Embedding dimension ({self.n_embd}) must be divisible by "
                f"the number of attention heads ({self.n_head})."
            )
