"""CUDA BF16 label readout for the merged Jet model."""
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, Qwen3_5ForCausalLM

def install_linear_attention():
    """Use FLA's differentiable chunk kernel; leave convolution on native PyTorch."""
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from transformers.models.qwen3_5 import modeling_qwen3_5 as impl
    def chunk(query, key, value, g, beta, chunk_size=64, initial_state=None,
              output_final_state=False, use_qk_l2norm_in_kernel=False, **kwargs):
        return chunk_gated_delta_rule(q=query,k=key,v=value,g=g,beta=beta,
            initial_state=initial_state,output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel)
    impl.torch_chunk_gated_delta_rule = chunk


def load_model(path):
    install_linear_attention()
    tokenizer = AutoTokenizer.from_pretrained(path)
    model, info = Qwen3_5ForCausalLM.from_pretrained(path, dtype=torch.bfloat16,
        device_map={'': 'cuda'}, attn_implementation='sdpa', output_loading_info=True)
    if info.get('missing_keys') or info.get('mismatched_keys') or info.get('unexpected_keys'):
        raise RuntimeError(f'Incomplete model load: {info}')
    model.config.use_cache = False
    model.eval()
    return model, tokenizer

def label_logits(model, example):
    base=model.get_base_model() if hasattr(model,'get_base_model') else model
    ids=torch.tensor([example['ids']],device='cuda',dtype=torch.long)
    hidden=base.model(input_ids=ids,use_cache=False).last_hidden_state[:, -1, :]
    labels=torch.tensor(example['labels'],device='cuda',dtype=torch.long)
    # Only materialize the requested rows of the frozen vocabulary head.
    return F.linear(hidden,base.lm_head.weight.index_select(0,labels))[0].float()

