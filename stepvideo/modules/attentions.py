import torch
import torch.nn as nn
from einops import rearrange

try:
    from xfuser.core.long_ctx_attention import xFuserLongContextAttention
except ImportError:
    xFuserLongContextAttention = None
    
    
class Attention(nn.Module):
    def __init__(self):
        super().__init__()
    
    def attn_processor(self, attn_type):
        """
        Returns the appropriate attention function based on the specified attention type.

        Args:
            attn_type (str): The type of attention to use. Supported types are 'torch' and 'parallel'.

        Returns:
            function: The corresponding attention function.

        Raises:
            Exception: If the specified attention type is not supported.
        """
        if attn_type == 'torch':
            return self.torch_attn_func
        elif attn_type == 'parallel':
            return self.parallel_attn_func
        else:
            raise Exception('Not supported attention type...')

    def torch_attn_func(
        self,
        q,
        k,
        v,
        attn_mask=None,
        causal=False,
        drop_rate=0.0,
        **kwargs
    ):
        """
        Computes attention using PyTorch's scaled dot-product attention.

        Args:
            q (torch.Tensor): Query tensor of shape (batch_size, seq_len, num_heads, head_dim).
            k (torch.Tensor): Key tensor of shape (batch_size, seq_len, num_heads, head_dim).
            v (torch.Tensor): Value tensor of shape (batch_size, seq_len, num_heads, head_dim).
            attn_mask (torch.Tensor, optional): Attention mask tensor. Defaults to None.
            causal (bool, optional): Whether to apply causal masking. Defaults to False.
            drop_rate (float, optional): Dropout rate. Defaults to 0.0.

        Returns:
            torch.Tensor: The output tensor after applying attention.
        """
        if attn_mask is not None and attn_mask.dtype != torch.bool:
            attn_mask = attn_mask.to(q.dtype)
            
        if attn_mask is not None and attn_mask.ndim == 3:   ## no head
            n_heads = q.shape[2]
            attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
        
        q, k, v = map(lambda x: rearrange(x, 'b s h d -> b h s d'), (q, k, v))
        x = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=drop_rate, is_causal=causal
        )
        x = rearrange(x, 'b h s d -> b s h d')
        return x        

    def parallel_attn_func(
        self,
        q,
        k,
        v,
        causal=False,
        **kwargs
    ):
        """
        Computes attention using a parallel attention mechanism.

        Args:
            q (torch.Tensor): Query tensor of shape (batch_size, seq_len, num_heads, head_dim).
            k (torch.Tensor): Key tensor of shape (batch_size, seq_len, num_heads, head_dim).
            v (torch.Tensor): Value tensor of shape (batch_size, seq_len, num_heads, head_dim).
            causal (bool, optional): Whether to apply causal masking. Defaults to False.

        Returns:
            torch.Tensor: The output tensor after applying parallel attention.

        Raises:
            AssertionError: If xFuserLongContextAttention is not imported.
        """
        assert xFuserLongContextAttention is not None, 'to use sequence parallel attention, xFuserLongContextAttention should be imported...'
        hybrid_seq_parallel_attn = xFuserLongContextAttention()
        x = hybrid_seq_parallel_attn(
            None, q, k, v, causal=causal
        )
        return x