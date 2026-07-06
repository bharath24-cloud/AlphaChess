import os
import random
import logging
import numpy as np
import torch


# =========================
# Logging
# =========================

def setup_logger(
    name="AlphaChess",
    log_file=None,
    level=logging.INFO
):
    """
    Create logger
    """

    logger = logging.getLogger(
        name
    )

    logger.setLevel(
        level
    )

    if not logger.handlers:

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - "
            "%(levelname)s - %(message)s"
        )

        console_handler = (
            logging.StreamHandler()
        )

        console_handler.setFormatter(
            formatter
        )

        logger.addHandler(
            console_handler
        )

        if log_file:

            os.makedirs(
                os.path.dirname(
                    log_file
                ),
                exist_ok=True
            )

            file_handler = (
                logging.FileHandler(
                    log_file
                )
            )

            file_handler.setFormatter(
                formatter
            )

            logger.addHandler(
                file_handler
            )

    return logger


# =========================
# Seed
# =========================

def set_seed(
    seed=42
):
    """
    Reproducibility
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(
            seed
        )
        torch.cuda.manual_seed_all(
            seed
        )

    torch.backends.cudnn.deterministic = (
        True
    )

    torch.backends.cudnn.benchmark = (
        False
    )


# =========================
# Device
# =========================

def get_device():
    """
    Auto device
    """

    return (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


# =========================
# Model Utils
# =========================

def count_parameters(
    model
):
    """
    Trainable parameter count
    """

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def save_checkpoint(
    model,
    optimizer,
    path,
    epoch=None,
    extra=None
):
    """
    Save model checkpoint
    """

    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )

    checkpoint = {
        "model_state_dict":
            model.state_dict(),
        "optimizer_state_dict":
            optimizer.state_dict()
    }

    if epoch is not None:
        checkpoint["epoch"] = epoch

    if extra:
        checkpoint["extra"] = extra

    torch.save(
        checkpoint,
        path
    )

    print(
        f"Checkpoint saved -> {path}"
    )


def load_checkpoint(
    model,
    optimizer,
    path,
    device="cpu"
):
    """
    Load checkpoint
    """

    checkpoint = torch.load(
        path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    if (
        optimizer
        and
        "optimizer_state_dict"
        in checkpoint
    ):
        optimizer.load_state_dict(
            checkpoint[
                "optimizer_state_dict"
            ]
        )

    epoch = checkpoint.get(
        "epoch",
        0
    )

    extra = checkpoint.get(
        "extra",
        {}
    )

    print(
        f"Loaded checkpoint -> {path}"
    )

    return epoch, extra


# =========================
# Tensor Utils
# =========================

def to_tensor(
    array,
    device="cpu",
    dtype=torch.float32
):
    """
    numpy -> tensor
    """

    return torch.tensor(
        array,
        dtype=dtype
    ).to(device)


def softmax(
    x
):
    """
    Stable softmax
    """

    x = x - np.max(
        x
    )

    exp = np.exp(
        x
    )

    return exp / (
        np.sum(exp)
        + 1e-8
    )


# =========================
# Training Stats
# =========================

class AverageMeter:
    """
    Running average tracker
    """

    def __init__(
        self
    ):
        self.reset()

    def reset(
        self
    ):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(
        self,
        val,
        n=1
    ):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = (
            self.sum
            /
            self.count
        )


# =========================
# Board Utilities
# =========================

def legal_moves_to_indices(
    board,
    move_encoder
):
    """
    Legal move indices
    """

    indices = []

    for move in board.legal_moves:

        idx = (
            move_encoder.move_to_index(
                move
            )
        )

        if idx is not None:
            indices.append(
                idx
            )

    return indices


# =========================
# Memory
# =========================

def gpu_memory():
    """
    GPU memory info
    """

    if not torch.cuda.is_available():
        return "CPU"

    allocated = (
        torch.cuda.memory_allocated()
        /
        1024**2
    )

    reserved = (
        torch.cuda.memory_reserved()
        /
        1024**2
    )

    return {
        "allocated_MB":
            round(
                allocated,
                2
            ),
        "reserved_MB":
            round(
                reserved,
                2
            )
    }


if __name__ == "__main__":

    logger = setup_logger()

    logger.info(
        "Utils loaded"
    )

    set_seed(42)

    print(
        "Device:",
        get_device()
    )

    print(
        "GPU Memory:",
        gpu_memory()
    )