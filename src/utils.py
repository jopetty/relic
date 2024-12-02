from torch.optim import AdamW

from transformers import get_scheduler
from transformers.trainer_pt_utils import get_parameter_names
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS


def create_optimizer_scheduler(
    model, lr, max_steps, optimizer_type, lr_scheduler_type, warmup_steps=50
):
    """
    Create AdamW optimizer and learning rate scheduler.
    """
    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if n not in decay_parameters
            ],
            "weight_decay": 0.00,
        },
    ]
    if optimizer_type == "adamw":
        optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-8)
    else:
        raise ValueError(f"Optimizer type {optimizer_type} not recognized.")

    scheduler = get_scheduler(
        name=lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps,
    )

    return optimizer, scheduler
