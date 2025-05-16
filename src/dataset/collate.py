import numpy as np
import torch


def custom_collate(batch):
    """
    Custom collate function that handles dictionaries with PIL.Image entries.
    """
    if isinstance(batch[0], dict):
        collated = {}
        for key in batch[0]:
            values = [item[key] for item in batch]
            if isinstance(values[0], torch.Tensor):
                collated[key] = torch.stack(values)
            elif isinstance(values[0], np.ndarray):
                collated[key] = torch.tensor(np.stack(values))
            else:
                collated[key] = values  # fallback: list of objects
        return collated
    else:
        raise TypeError(f"Unsupported batch type: {type(batch[0])}")
