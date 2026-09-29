import torch


def masked_mse_loss(prediction, target):

    mask = torch.isfinite(target)

    safe_target = torch.where(
        mask,
        target,
        torch.zeros_like(target)
    )

    squared_error = (
        prediction - safe_target
    ) ** 2

    squared_error = squared_error[mask]

    if squared_error.numel() == 0:
        return torch.tensor(
            0.0,
            device=prediction.device,
            requires_grad=True
        )

    return squared_error.mean()