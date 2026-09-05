from pathlib import Path
import numpy as np
import torch


def save_support_checkpoint(
    model,
    threshold,
    path,
    metadata=None,
):
    """
    Save everything required to reproduce TorchMLP inference:
      - network weights
      - feature normalization
      - validation-selected threshold
      - architecture metadata
      - experiment metadata
    """

    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if model.net is None:
        raise RuntimeError(
            "Cannot save uninitialized support model"
        )

    if model.mean_ is None or model.std_ is None:
        raise RuntimeError(
            "Missing MLP normalization statistics"
        )

    state_dict = {
        k: v.detach().cpu()
        for k, v in model.net.state_dict().items()
    }

    ckpt = {
        "format_version": 1,

        "model_type":
            "TorchMLP_support_classifier",

        "architecture": {
            "input_dim":
                int(len(model.mean_)),
            "hidden_1": 64,
            "hidden_2": 32,
            "dropout": 0.10,
            "output_dim": 1,
        },

        "state_dict":
            state_dict,

        "mean":
            np.asarray(
                model.mean_,
                dtype=np.float32,
            ),

        "std":
            np.asarray(
                model.std_,
                dtype=np.float32,
            ),

        "threshold":
            float(threshold),

        "metadata":
            {} if metadata is None
            else dict(metadata),
    }

    torch.save(
        ckpt,
        path,
    )

    print(
        "Saved support checkpoint:",
        path
    )

    return path


def load_support_checkpoint(
    path,
    model_cls,
    device=None,
):
    """
    Reconstruct the fixed 14→64→32→1 support MLP.
    """

    path = Path(path)

    if device is None:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    else:
        device = torch.device(
            device
        )

    ckpt = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    arch = ckpt[
        "architecture"
    ]

    model = model_cls()

    model.mean_ = np.asarray(
        ckpt["mean"],
        dtype=np.float32,
    )

    model.std_ = np.asarray(
        ckpt["std"],
        dtype=np.float32,
    )

    model.net = torch.nn.Sequential(
        torch.nn.Linear(
            arch["input_dim"],
            arch["hidden_1"],
        ),
        torch.nn.ReLU(),
        torch.nn.Dropout(
            p=arch["dropout"],
        ),

        torch.nn.Linear(
            arch["hidden_1"],
            arch["hidden_2"],
        ),
        torch.nn.ReLU(),
        torch.nn.Dropout(
            p=arch["dropout"],
        ),

        torch.nn.Linear(
            arch["hidden_2"],
            arch["output_dim"],
        ),
    ).to(device)

    model.net.load_state_dict(
        ckpt["state_dict"]
    )

    model.net.eval()

    threshold = float(
        ckpt["threshold"]
    )

    return (
        model,
        threshold,
        ckpt.get(
            "metadata",
            {},
        ),
    )
