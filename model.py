import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional
import sentencepiece
import tqdm
from triton.language import multiple_of

# Note: This is a simplified version of the model, it only supports one token at a time, and it only supports the forward pass.
# It support inference only. For training one needs to remove the KV cache and the freqs_complex.

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32 # Number of blocks in the decoder 
    n_heads: int = 32 # Number of heads for Queries
    n_kv_heads: Optional[int] = None # Number of heads for Key/Value queries
    vocab_size: int = -1 # Default to -1, which means using sentencepiece to get the vocab size, Will be set from tokenizer
    multiple_of: int = 256 # Will be used to round the hidden dim to the nearest multiple
    ffn_dim_multiplier: Optional[float] = None # Will be used to scale the ffn dim (only used to make the number of parameters
    # similar to vanilla transformer for comparison)
    norm_eps: float = 1e-5 # Epsilon for layer norm

    # For KV cache
    max_batch_size: int = 32 # Maximum batch size for KV cache
    max_seq_len: int = 2048 # Maximum sequence length for KV cache

    device: str = None # Device to use for training

def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta: float = 10000.0) -> torch.Tensor:
    # Mentioned in the paper, dimensions of the embeddings must be even
    assert head_dim % 2 == 0, "Dimensions of the embeddings must be even"
    # Build the theta parameter
    # According to the formula in paper: theta_1 = 10000 ^ (-2(i-1/dim) for i = [1, 2, ...dim / 2]
    # shape:  (Head_Dim / 2)
    theta_numerator = torch.arange(0, head_dim, 2).float()
    # Shape: (Head_Dim / 2)
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device)

    # Build the positions (the "m" parameter in the paper)
    # Shape: (Seq_len)
    m = torch.arange(seq_len, device=device).float()
    # Multiply each theta by each position m to get the frequencies
    # (Seq_len) x (Head_Dim / 2) -> (Seq_len, Head_Dim / 2)  # Note "X" is the outer product
    freqs = torch.outer(m, theta)
    # We can compute complex numbers in the polar form c = R * e^(i * m * theta), where R = 1 as follows:
    # (Seq_len, Head_Dim / 2) -> (Seq_len, Head_Dim / 2)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)

    return freqs_complex

def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
    # (B, Seq_Len, H, Head_Dim) -> (B, Seq_len, H, Head_Dim / 2) # Note: "Head_Dim / 2" because all 2 consecutive dims are forming a complex number 
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # (Seq_len, Head_Dim / 2) --> (1, Seq_len, 1, Head_Dim / 2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    # (B, Seq_len, H, Head_Dim / 2) * (1, Seq_len, 1, Head_Dim / 2)--> (B, Seq_len, H, Head_Dim / 2)
    x_rotated = x_complex * freqs_complex

    # (B, Seq_len, H, Head_Dim / 2) --> (B, Seq_len, H, Head_Dim / 2, 2)
    x_out = torch.view_as_real(x_rotated)
    # (B, Seq_len, H, Head_Dim / 2, 2) --> (B, Seq_len, H, Head_Dim)
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    else:
        return (
            # (B, Seq_len, N_KV_Heads, 1, Head_Dim)
            x[:,:,:, None, :]
            .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
            .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
        )

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # Gamma param
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        # (B, Seq_len, Dim) * (B, Seq_len, 1) --> (B, Seq_len, Dim)
        # rsqrt: 1/sqrt(x)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        # (Dim) * (B, Seq_len, Dim) --> (B, Seq_len, Dim)
        return self.weight * self._norm(x.float()).type_as(x)

class EncoderBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads

        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)

        # Normalization Before the self attention
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        # Normalization before the feed forward block
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        # (B, Seq_len, Dim) + (B, Seq_len, Dim) --> (B, Seq_len, Dim)
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_complex)
        out = h +  self.feed_forward.forward(self.ffn_norm(h))
        return out

class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        # The number of heads for Keys and Values
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # The nzmber of heads for the Query
        self.n_heads_q = args.n_heads
        # Indicates how many times the heads of Keys and Values should be repeated to match the head of the Queries
        self.n_rep = self.n_heads_q // self.n_kv_heads
        # Dim of each head
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias = False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias = False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias = False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias = False)

        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        batch_size, seq_len, _ = x.shape # (B, Seq_len, Dim) # We know Seq_len = 1

        # (B, 1, Dim) --> (B, 1, H_Q * Head_Dim )
        xq = self.wq(x)
        # (B, 1, Dim) --> (B, H_KV * Head_Dim ) # Note Here the H_KV * Head_Dim may be different to Dim depending on Head_Dim
        xk = self.wk(x)
        xv = self.wv(x)

        # (B, 1, H_Q * Head_Dim) --> (B, 1, H_Q, Head_Dim)
        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)

        # (B, 1, H_KV * Head_Dim) --> (B, 1, H_KV, Head_Dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # Apply rotary positional embeddings # Note it does not change the dimensions of the tensor
        xq = apply_rotary_embeddings(xq, freqs_complex, device= x.device)
        xk = apply_rotary_embeddings(xk, freqs_complex, device= x.device)

        # Replace the entry in the caches for this token
        self.cache_k[:batch_size, start_pos: start_pos + seq_len] = xk
        self.cache_v[:batch_size, start_pos: start_pos + seq_len] = xv

        # Get all the cached keys and values so far
        # (B, seq_len_kv, H_KV, Head_Dim)
        keys = self.cache_k[:batch_size, 0:start_pos + seq_len]
        values = self.cache_v[:batch_size, 0:start_pos + seq_len]

        # Repeat the heads of the K and V to reach the number of heads of the queries
        ##NOTE: Here we repeat tge keys and values corresponding to the head of the queries
        # If n_heads_q = 8  and n_heeads_kv = 4 then we repeat the same heads 2 times so that there are one matrix of K and V per Q
        # This somewhat defeats the logic of having less heads of K and V for Grouped query attention but in LLAMA repo they only provide
        # We have less parameters to train as we just repeat the heads of K and V.
        # Grouped query attention for 70B model which I can not test thus for this small model I follow the repeating strategy
        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        # (B, 1, H_Q, Head_dim) --> (B, H_Q, 1 , Head_Dim)
        xq = xq.transpose(1,2)
        keys = keys.transpose(1,2)
        values = values.transpose(1,2)

        # (B, H_Q, 1, Head_dim) @ (B, H_Q, Head_Dim, Seq_len_KV) --> (B, H_Q, 1, Seq_len_KV)
        scores = torch.matmul(xq, keys.transpose(2,3)) / math.sqrt(self.head_dim)
        scores  = F.softmax(scores.float(), dim=-1).type_as(xq)

        # (B, H_Q, 1, Seq_Len) @ (B, H_Q, Seq_Len_KV, Head_Dim) --> (B, H_Q, 1, Head_Dim)
        output = torch.matmul(scores, values)

        # (B, H_Q, 1, Head_Dim) -> (B, 1, H_Q, Head_Dim) -> (B, 1, Dim )
        output = (output.transpose(1,2).contiguous().view(batch_size, seq_len, -1))
        return self.wo(output) # (B, 1, Dim) -> (B, 1, Dim)

class FeedForward(nn.Module):
        def __init__(self, args: ModelArgs):
            super().__init__()

            hidden_dim = 4 * args.dim
            hidden_dim = int(2 * hidden_dim / 3)
            if args.ffn_dim_multiplier is not None:
                hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
            # Round the hidden_dim to the nearest multiple of the multiplier_of parameter

            hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of -1) // args.multiple_of)

            self.w1 = nn.Linear(args.dim, hidden_dim, bias = False)
            self.w2 = nn.Linear(hidden_dim, args.dim, bias = False)
            self.w3 = nn.Linear(args.dim, hidden_dim, bias = False)


        def forward(self, x: torch.Tensor):
            swish = F.silu(self.w1(x))
            x_V = self.w3(x)
            x = swish * x_V
            x = self.w2(x)
            return x
            

class Transformer(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        # Making sure vocab size is set
        assert args.vocab_size != -1, "vocab_size must be set"

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)

        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(EncoderBlock(args))

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)

        self.freqs_complex = precompute_theta_pos_frequencies(self.args.dim // self.args.n_heads, self.args.max_seq_len * 2, device=self.args.device)

    def forward(self, tokens: torch.Tensor, start_pos: int) -> torch.Tensor:
        # (B, Seq_len)
        batch_size, seq_len = tokens.shape
        assert seq_len == 1, "Only one token is supported at a time"

        # (B, Seq_len) -> (B, Seq_len, dim)
        h = self.tok_embeddings(tokens)

        # Retrieve the pairs (m, theta) corresponding to the positions [start_pos, start_pos + seq_len]
        freqs_complex = self.freqs_complex[start_pos:start_pos + seq_len]

        # Consecutively apply all the encoder blocks
        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)
        
        h = self.norm(h)

        # (B, Seq_len, dim) -> (B, Seq_len, vocab_size)
        logits = self.output(h).float()

        return logits


