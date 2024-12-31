import torch
from transformers import AutoModelForCausalLM
import numpy as np
from typing import Union
from copy import deepcopy

import torch
import numpy as np
from typing import Union
from copy import deepcopy


import torch
import numpy as np
from typing import Union
from copy import deepcopy


def maybe_torch(tensor, is_torch, ref_tensor):
    """
    Convert a tensor back to torch if necessary, preserving device placement.

    Args:
        tensor: Numpy array to potentially convert
        is_torch: Whether to convert to torch tensor
        ref_tensor: Reference tensor for device placement

    Returns:
        torch.Tensor or np.ndarray: Converted tensor on correct device
    """
    if is_torch:
        return torch.from_numpy(tensor).to(
            ref_tensor.device if hasattr(ref_tensor, "device") else "cpu"
        )
    return tensor


def slerp(
    t: Union[float, np.ndarray],
    v0: Union[np.ndarray, torch.Tensor],
    v1: Union[np.ndarray, torch.Tensor],
    DOT_THRESHOLD: float = 0.9995,
    eps: float = 1e-8,
):
    """
    Spherical linear interpolation between two vectors.

    Args:
        t (float/np.ndarray): Interpolation factor between 0.0 and 1.0
        v0 (np.ndarray/torch.Tensor): Starting vector
        v1 (np.ndarray/torch.Tensor): Final vector
        DOT_THRESHOLD (float): Threshold for considering vectors colinear
        eps (float): Small value to prevent division by zero

    Returns:
        Union[np.ndarray, torch.Tensor]: Interpolated vector
    """
    # Track if inputs are torch tensors
    is_torch = torch.is_tensor(v0)

    # Convert to numpy arrays if needed
    if is_torch:
        v0 = v0.detach().cpu().float().numpy()
        v1 = v1.detach().cpu().float().numpy()

    # Store original vectors
    v0_orig = v0.copy()
    v1_orig = v1.copy()

    # Normalize vectors
    v0_norm = v0 / (np.linalg.norm(v0) + eps)
    v1_norm = v1 / (np.linalg.norm(v1) + eps)

    # Calculate dot product
    dot = np.sum(v0_norm * v1_norm)

    # Handle nearly parallel vectors
    if np.abs(dot) > DOT_THRESHOLD:
        result = (1.0 - t) * v0_orig + t * v1_orig
        return maybe_torch(result, is_torch, v0_orig)

    # SLERP formula
    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))  # Clip to prevent numerical errors
    sin_theta_0 = np.sin(theta_0)

    theta_t = theta_0 * t
    sin_theta_t = np.sin(theta_t)

    s0 = np.sin(theta_0 - theta_t) / sin_theta_0
    s1 = sin_theta_t / sin_theta_0

    result = s0 * v0_orig + s1 * v1_orig

    # Convert back to torch if needed using maybe_torch
    return maybe_torch(result, is_torch, v0_orig)


def merge_models_slerp(
    model1: torch.nn.Module, model2: torch.nn.Module, alpha: float
) -> torch.nn.Module:
    """
    Merge two PyTorch models using SLERP interpolation with proper handling of different layer types.

    Args:
        model1: First model
        model2: Second model
        alpha: Interpolation factor (0.0 = model1, 1.0 = model2)

    Returns:
        Merged model
    """
    if not 0 <= alpha <= 1:
        raise ValueError("Alpha must be between 0 and 1")

    merged_model = deepcopy(model1)

    for (name1, module1), (name2, module2) in zip(
        model1.named_modules(), model2.named_modules()
    ):
        if name1 != name2:
            raise ValueError(f"Module names don't match: {name1} vs {name2}")

        print("Merging module:", name1)

        merged_module = merged_model.get_submodule(name1)

        # Handle different types of layers
        if isinstance(module1, (torch.nn.Linear, torch.nn.Embedding)):
            # Merge weights
            merged_module.weight.data = slerp(
                alpha, module1.weight.data, module2.weight.data
            )

            # Merge biases if they exist
            if getattr(module1, "bias", None) is not None:
                merged_module.bias.data = slerp(
                    alpha, module1.bias.data, module2.bias.data
                )

        elif isinstance(module1, torch.nn.LayerNorm):
            # For LayerNorm, we need to handle weight, bias, and running stats
            merged_module.weight.data = slerp(
                alpha, module1.weight.data, module2.weight.data
            )
            if module1.bias is not None:
                merged_module.bias.data = slerp(
                    alpha, module1.bias.data, module2.bias.data
                )

            # Properly handle eps parameter
            merged_module.eps = (1 - alpha) * module1.eps + alpha * module2.eps
        else:
            print("Skipping module:", name1)

    return merged_model


# def merge_models_slerp(
#     model1: torch.nn.Module, model2: torch.nn.Module, alpha: float
# ) -> torch.nn.Module:
#     """
#     Merge two PyTorch models using SLERP interpolation.

#     Args:
#         model1: First model
#         model2: Second model
#         alpha: Interpolation factor (0.0 = model1, 1.0 = model2)

#     Returns:
#         Merged model
#     """
#     if not 0 <= alpha <= 1:
#         raise ValueError("Alpha must be between 0 and 1")

#     merged_model = deepcopy(model1)

#     # Get parameter names that exist in both models
#     params1 = dict(model1.named_parameters())
#     params2 = dict(model2.named_parameters())
#     shared_params = set(params1.keys()) & set(params2.keys())

#     for name in shared_params:
#         param1 = params1[name]
#         param2 = params2[name]

#         if param1.shape != param2.shape:
#             raise ValueError(
#                 f"Parameter shapes don't match for {name}: {param1.shape} vs {param2.shape}"
#             )

#         # Get parameter in merged model
#         merged_param = merged_model.get_parameter(name)

#         # Apply SLERP
#         merged_param.data = slerp(alpha, param1.data, param2.data)

#     return merged_model


# def slerp(
#     t: Union[float, np.ndarray],
#     v0: Union[np.ndarray, torch.Tensor],
#     v1: Union[np.ndarray, torch.Tensor],
#     DOT_THRESHOLD: float = 0.9995,
#     eps: float = 1e-8,
# ):
#     """
#     Spherical linear interpolation

#     From: https://gist.github.com/dvschultz/3af50c40df002da3b751efab1daddf2c
#     Args:
#         t (float/np.ndarray): Float value between 0.0 and 1.0
#         v0 (np.ndarray): Starting vector
#         v1 (np.ndarray): Final vector
#         DOT_THRESHOLD (float): Threshold for considering the two vectors as
#                                colinear. Not recommended to alter this.
#     Returns:
#         v2 (np.ndarray): Interpolation vector between v0 and v1
#     """
#     is_torch = False
#     if not isinstance(v0, np.ndarray):
#         is_torch = True
#         v0 = v0.detach().cpu().float().numpy()
#     if not isinstance(v1, np.ndarray):
#         is_torch = True
#         v1 = v1.detach().cpu().float().numpy()

#     # Copy the vectors to reuse them later
#     v0_copy = np.copy(v0)
#     v1_copy = np.copy(v1)

#     # Normalize the vectors to get the directions and angles
#     v0 = normalize(v0, eps)
#     v1 = normalize(v1, eps)

#     # Dot product using element-wise multiplication and sum
#     dot = np.sum(v0 * v1)

#     # If absolute value of dot product is almost 1, vectors are ~colinear, so use lerp
#     if np.abs(dot) > DOT_THRESHOLD:
#         res = lerp(t, v0_copy, v1_copy)
#         return maybe_torch(res, is_torch, v0_copy)

#     # Calculate initial angle between v0 and v1
#     theta_0 = np.arccos(dot)
#     sin_theta_0 = np.sin(theta_0)

#     # Angle at timestep t
#     theta_t = theta_0 * t
#     sin_theta_t = np.sin(theta_t)

#     # Finish the slerp algorithm
#     s0 = np.sin(theta_0 - theta_t) / sin_theta_0
#     s1 = sin_theta_t / sin_theta_0
#     res = s0 * v0_copy + s1 * v1_copy

#     return maybe_torch(res, is_torch, v0_copy)


# def lerp(t, v0, v1):
#     """Linear interpolation"""
#     return (1.0 - t) * v0 + t * v1


# def normalize(v, eps=1e-8):
#     """Normalize a vector"""
#     return v / (np.linalg.norm(v) + eps)  # Avoid division by zero


# def maybe_torch(tensor, is_torch, ref_tensor):
#     """Convert a tensor back to torch if necessary"""
#     if is_torch:
#         return torch.from_numpy(tensor).to(
#             ref_tensor.device if hasattr(ref_tensor, "device") else "cpu"
#         )  # Maintain original device
#     else:
#         return tensor


# def merge_models_slerp(model1, model2, alpha):
#     """Merges two Hugging Face models using SLERP."""

#     # keep unmergeable parts from model1, like dropout
#     merged_model = deepcopy(model1)

#     for (name1, module1), (name2, module2) in zip(
#         model1.named_modules(), model2.named_modules()
#     ):
#         if name1 != name2:
#             raise ValueError(f"Module names don't match: {name1} vs {name2}")

#         # print(f"Merging module: {name1}")

#         if isinstance(module1, torch.nn.Linear):  # Apply to linear layers
#             v0 = module1.weight.data
#             v1 = module2.weight.data
#             merged_weight = slerp(alpha, v0, v1)
#             merged_bias = (
#                 slerp(alpha, module1.bias.data, module2.bias.data)
#                 if module1.bias is not None
#                 else None
#             )

#             merged_module = merged_model.get_submodule(name1)
#             merged_module.weight = torch.nn.Parameter(merged_weight)
#             if merged_bias is not None:
#                 merged_module.bias = torch.nn.Parameter(merged_bias)
#         elif isinstance(module1, torch.nn.LayerNorm):  # Apply to layernorm
#             v0 = module1.weight.data
#             v1 = module2.weight.data
#             merged_weight = slerp(alpha, v0, v1)
#             merged_bias = (
#                 slerp(alpha, module1.bias.data, module2.bias.data)
#                 if module1.bias is not None
#                 else None
#             )

#             merged_module = merged_model.get_submodule(name1)
#             merged_module.weight = torch.nn.Parameter(merged_weight)
#             if merged_bias is not None:
#                 merged_module.bias = torch.nn.Parameter(merged_bias)
#         elif isinstance(module1, torch.nn.Embedding):  # Apply to embeddings
#             v0 = module1.weight.data
#             v1 = module2.weight.data
#             merged_weight = slerp(alpha, v0, v1)

#             merged_module = merged_model.get_submodule(name1)
#             merged_module.weight = torch.nn.Parameter(merged_weight)
#         # else:
#         #     print(f"Skipping module: {name1}")

#     return merged_model


def main(output_dir, model_name1, model_name2, alpha=0.5):
    model1 = AutoModelForCausalLM.from_pretrained(model_name1)
    model2 = AutoModelForCausalLM.from_pretrained(model_name2)

    merged_model = merge_models_slerp(model1, model2, alpha)

    # Save the merged model
    merged_model.save_pretrained(output_dir)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
