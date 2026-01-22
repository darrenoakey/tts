import torch
from src.flash_attn_shim import flash_attn_varlen_func, install_flash_attn_shim


# ##################################################################
# test flash attn varlen func basic
# verify the shim function produces output of correct shape
def test_flash_attn_varlen_func_basic():
    seq_len = 10
    n_heads = 4
    head_dim = 16

    q = torch.randn(seq_len, n_heads, head_dim)
    k = torch.randn(seq_len, n_heads, head_dim)
    v = torch.randn(seq_len, n_heads, head_dim)
    cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32)

    out = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, seq_len, seq_len)

    assert out.shape == (seq_len, n_heads, head_dim)


# ##################################################################
# test flash attn varlen func multiple sequences
# verify the shim handles multiple variable length sequences
def test_flash_attn_varlen_func_multiple_sequences():
    n_heads = 4
    head_dim = 16
    seq_lens = [5, 8, 3]
    total_len = sum(seq_lens)

    q = torch.randn(total_len, n_heads, head_dim)
    k = torch.randn(total_len, n_heads, head_dim)
    v = torch.randn(total_len, n_heads, head_dim)

    cu_seqlens = torch.tensor([0] + list(torch.cumsum(torch.tensor(seq_lens), 0)), dtype=torch.int32)

    out = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max(seq_lens), max(seq_lens))

    assert out.shape == (total_len, n_heads, head_dim)


# ##################################################################
# test install flash attn shim
# verify the shim installs fake flash_attn module
def test_install_flash_attn_shim():
    import sys
    install_flash_attn_shim()

    assert "flash_attn" in sys.modules
    assert "flash_attn.flash_attn_interface" in sys.modules

    from flash_attn.flash_attn_interface import flash_attn_varlen_func as imported_func
    assert imported_func is not None


# ##################################################################
# flash attn shim tests
# verify the flash attention shim provides compatible implementation
# for systems without cuda flash-attn package
