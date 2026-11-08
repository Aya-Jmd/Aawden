"""Check saved Aya adapter tensors against the real architecture without loading base weights."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from security_wsp.artifacts import atomic_json, file_hash, read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-config', type=Path, required=True)
    parser.add_argument('--base-config-sha256', default='81b7c1977e7c57052496bcbd50e94837db24e8188a5dc98efa0f9518fda9de7a')
    parser.add_argument('--config', type=Path, default=Path('configs/model-adapted.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if file_hash(args.base_config) != args.base_config_sha256:
        raise ValueError('The pinned base architecture configuration changed')
    specification = read_json(args.config)['lora_adapter']
    checkpoint = Path(specification['checkpoint'])
    if file_hash(checkpoint) != specification['sha256']:
        raise ValueError('The saved adapter changed')
    import torch
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM
    from safetensors import safe_open
    from security_wsp.adaptation import attach
    config = AutoConfig.from_pretrained(args.base_config.parent, local_files_only=True)
    with torch.device('meta'):
        model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16)
        expected = attach(model, specification['rank'], specification['target_modules'])
    count = 0
    with safe_open(checkpoint, framework='pt', device='cpu') as saved:
        if set(saved.keys()) != set(expected):
            raise ValueError('Adapter tensor names do not match the pinned architecture')
        for name, parameter in expected.items():
            tensor = saved.get_tensor(name)
            if tensor.shape != parameter.shape or tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
                raise ValueError('Invalid adapter tensor shape, precision or values')
            count += tensor.numel()
    result = {'passed': True, 'adapter_sha256': specification['sha256'],
              'base_config_sha256': args.base_config_sha256, 'tensors': len(expected), 'parameters': count,
              'torch': torch.__version__, 'transformers': transformers.__version__,
              'base_weights_loaded': False, 'cuda_used': False, 'behavioural_reload_verified': False,
              'scope': 'Fresh-process CPU tensor/architecture compatibility. No inference-equivalence claim.'}
    atomic_json(args.output, result)
    print('Verified', len(expected), 'adapter tensors against pinned Aya architecture; no GPU used')


if __name__ == '__main__':
    main()
