import json, struct, sys
import numpy as np
from gguf import GGUFWriter, GGUFValueType, GGMLQuantizationType

SRC = 'models/qwen25-coder-1.5b/model.safetensors'
OUT = sys.argv[1] if len(sys.argv) > 1 else 'models/qwen25-coder-1.5b/synth-f16.gguf'
QUANT = len(sys.argv) > 2 and sys.argv[2] == 'q40'

with open(SRC, 'rb') as f:
    hlen = struct.unpack('<Q', f.read(8))[0]
    hdr = json.loads(f.read(hlen))
    data_base = 8 + hlen
    tensors = {}
    for name, info in hdr.items():
        if name == '__metadata__':
            continue
        f.seek(data_base + info['data_offsets'][0])
        n = int(np.prod(info['shape']))
        if info['dtype'] == 'BF16':
            u = np.frombuffer(f.read(n * 2), dtype=np.uint16).astype(np.uint32)
            tensors[name] = ((u << 16).view(np.float32)).reshape(info['shape'])
        elif info['dtype'] == 'F32':
            tensors[name] = np.frombuffer(f.read(n * 4), dtype=np.float32).reshape(info['shape'])
        elif info['dtype'] == 'F16':
            tensors[name] = np.frombuffer(f.read(n * 2), dtype=np.float16).astype(np.float32).reshape(info['shape'])
        else:
            raise ValueError(info['dtype'])

def load(name):
    return tensors[name]

HIDDEN, INTER, LAYERS, VHEADS = 1536, 8960, 28, 12
KV = 2 * (HIDDEN // VHEADS)  # kv_dim = 256
HD = HIDDEN // VHEADS       # 128

w = GGUFWriter(OUT, arch='qwen2')
w.add_embedding_length(HIDDEN)
w.add_block_count(LAYERS)
w.add_feed_forward_length(INTER)
w.add_head_count(VHEADS)
w.add_head_count_kv(2)
w.add_context_length(32768)
w.add_rope_freq_base(1000000.0)
w.add_layer_norm_rms_eps(1e-6)

def add(name, arr, qtype=None):
    a = np.ascontiguousarray(arr)
    if QUANT and qtype is not None and a.ndim == 2:
        w.add_tensor(name, a.astype(np.float32), raw_dtype=qtype)
    else:
        w.add_tensor(name, a.astype(np.float16))

add('token_embd.weight', load('model.embed_tokens.weight'))
add('output_norm.weight', load('model.norm.weight'))
# no output.weight: tied embeddings, loader falls back to token_embd

for l in range(LAYERS):
    p = f'model.layers.{l}.'
    g = f'blk.{l}.'
    add(g + 'attn_norm.weight', load(p + 'input_layernorm.weight'))
    add(g + 'attn_q.weight',   load(p + 'self_attn.q_proj.weight'))
    add(g + 'attn_k.weight',   load(p + 'self_attn.k_proj.weight'))
    add(g + 'attn_v.weight',   load(p + 'self_attn.v_proj.weight'))
    add(g + 'attn_q.bias',     load(p + 'self_attn.q_proj.bias'))
    add(g + 'attn_k.bias',     load(p + 'self_attn.k_proj.bias'))
    add(g + 'attn_v.bias',     load(p + 'self_attn.v_proj.bias'))
    add(g + 'attn_output.weight', load(p + 'self_attn.o_proj.weight'))
    add(g + 'ffn_norm.weight', load(p + 'post_attention_layernorm.weight'))
    add(g + 'ffn_gate.weight', load(p + 'mlp.gate_proj.weight'))
    add(g + 'ffn_up.weight',   load(p + 'mlp.up_proj.weight'))
    add(g + 'ffn_down.weight', load(p + 'mlp.down_proj.weight'))

w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print('wrote', OUT)
