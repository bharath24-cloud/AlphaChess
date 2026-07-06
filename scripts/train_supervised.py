import os
import sys
import torch

# Add project root
sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

from core.trainer import AlphaChessTrainer


def main():

    # Auto GPU / CPU
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Using device: {device}"
    )

    # Paths
    pgn_path = (
        "data/lichess_games.pgn"
    )

    save_path = (
        "saved_models/alphachess_v1.pth"
    )

    # Training settings
    epochs = 5
    max_games = 300000
    batch_size = 64
    learning_rate = 1e-3
    streaming = True

    trainer = AlphaChessTrainer(
        lr=learning_rate,
        batch_size=batch_size,
        device=device
    )

    # Resume if checkpoint exists
    if os.path.exists(
        save_path
    ):
        print(
            "Checkpoint found."
        )

        choice = input(
            "Resume training? (y/n): "
        ).lower()

        if choice == "y":
            trainer.load_model(
                save_path
            )

    trainer.train_from_pgn(
        pgn_path=pgn_path,
        epochs=epochs,
        max_games=max_games,
        streaming=streaming,
        save_path=save_path
    )

    print(
        "Supervised training complete."
    )


if __name__ == "__main__":
    main()