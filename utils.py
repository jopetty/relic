import pickle
import torch


def freeze_all_expecting_pruning_params(model):
    for n, p in model.named_parameters():
        if "log_alpha" in n or "sparsity_lambda" in n:
            p.requires_grad = True
        else:
            p.requires_grad = False


def load_avg_activations(model, avg_activation_path, device):
    avg_activations = pickle.load(open(avg_activation_path, "rb"))
    for n, m in model.named_modules():
        n = ".".join(n.split(".")[1:])
        if n in avg_activations:
            activation = (
                torch.from_numpy(avg_activations[n]).to(device).to(torch.float32)
            )
            m.set_avg_activation(activation)
