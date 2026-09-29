import torch


def masked_mse_loss(prediction, target):

    # True where target is valid
    mask = torch.isfinite(target)

    # Replace invalid target values temporarily
    safe_target = torch.where(
        mask,
        target,
        torch.zeros_like(target)
    )

    # Calculate squared error
    squared_error = (
        prediction - safe_target
    ) ** 2

    # Keep only valid pixels
    squared_error = squared_error[mask]

    # Avoid empty mask
    if squared_error.numel() == 0:
        return torch.tensor(
            0.0,
            device=prediction.device,
            requires_grad=True
        )

    return squared_error.mean()