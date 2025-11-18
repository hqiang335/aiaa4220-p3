import torch

# Load both checkpoints
original = torch.load("evaluation_mini_exp1/falcon/hm3d/checkpoints/ckpt.19.pth")
processed = torch.load("evaluation_mini_exp1/falcon/hm3d/checkpoints/eval.pth")

original_keys = len(original[0]['state_dict'])
processed_keys = len(processed[0]['state_dict'])

print(f"Original checkpoint: {original_keys} keys")
print(f"Processed checkpoint: {processed_keys} keys")
print(f"Removed: {original_keys - processed_keys} keys")

# Check for aux_loss_modules
aux_keys = [k for k in original[0]['state_dict'].keys() if k.startswith('aux_loss_modules')]
print(f"\nAuxiliary modules removed: {len(aux_keys)}")