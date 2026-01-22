import sys
import types
import torch
import torch.nn.functional as F


# ##################################################################
# flash attn varlen func shim
# provides a compatible implementation of flash_attn_varlen_func using pytorch sdpa
def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p=0.0):
    # ##################################################################
    # flash attn varlen func shim
    # process variable length sequences using pytorch scaled dot product attention
    _ = max_seqlen_q, max_seqlen_k
    batch_size = len(cu_seqlens_q) - 1
    outputs = []

    for i in range(batch_size):
        start_q = cu_seqlens_q[i].item()
        end_q = cu_seqlens_q[i + 1].item()
        start_k = cu_seqlens_k[i].item()
        end_k = cu_seqlens_k[i + 1].item()

        q_seq = q[start_q:end_q]
        k_seq = k[start_k:end_k]
        v_seq = v[start_k:end_k]

        q_t = q_seq.transpose(0, 1).unsqueeze(0)
        k_t = k_seq.transpose(0, 1).unsqueeze(0)
        v_t = v_seq.transpose(0, 1).unsqueeze(0)

        out = F.scaled_dot_product_attention(q_t, k_t, v_t, dropout_p=dropout_p)

        out = out.squeeze(0).transpose(0, 1)
        outputs.append(out)

    return torch.cat(outputs, dim=0)


# ##################################################################
# install shim
# inject fake flash_attn module into sys.modules before qwen-tts imports it
def install_flash_attn_shim():
    # ##################################################################
    # install shim
    # create proper module objects with spec and inject into sys.modules
    from importlib.machinery import ModuleSpec

    flash_attn = types.ModuleType("flash_attn")
    flash_attn.__spec__ = ModuleSpec("flash_attn", None)
    flash_attn.__package__ = "flash_attn"
    flash_attn.__path__ = []

    flash_attn_interface = types.ModuleType("flash_attn.flash_attn_interface")
    flash_attn_interface.__spec__ = ModuleSpec("flash_attn.flash_attn_interface", None)
    flash_attn_interface.__package__ = "flash_attn"

    flash_attn_interface.flash_attn_varlen_func = flash_attn_varlen_func
    flash_attn_interface.flash_attn_unpadded_func = flash_attn_varlen_func

    flash_attn.flash_attn_interface = flash_attn_interface

    sys.modules["flash_attn"] = flash_attn
    sys.modules["flash_attn.flash_attn_interface"] = flash_attn_interface


# ##################################################################
# flash attn shim module
# provides a pytorch-based implementation of flash attention for systems
# without cuda, allowing qwen-tts to use accelerated attention on apple silicon
