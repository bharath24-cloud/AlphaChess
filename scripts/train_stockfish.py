import os
import sys
import chess
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

from core.stockfish_trainer import StockfishTrainer


def generate_boards(
    num_positions=500
):
    """
    Generate training positions
    from random legal play
    """

    boards = []

    for _ in range(num_positions):

        board = chess.Board()

        moves_to_play = torch.randint(
            5,
            30,
            (1,)
        ).item()

        for _ in range(
            moves_to_play
        ):

            if board.is_game_over():
                break

            legal_moves = list(
                board.legal_moves
            )

            move = legal_moves[
                torch.randint(
                    len(legal_moves),
                    (1,)
                ).item()
            ]

            board.push(
                move
            )

        if not board.is_game_over():
            boards.append(
                board.copy()
            )

    return boards


def main():

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Using device: {device}"
    )

    # Update this path
    stockfish_path = (
        "stockfish/stockfish-windows-x86-64-avx2.exe"
    )

    save_path = (
        "saved_models/alphachess_stockfish.pth"
    )

    trainer = StockfishTrainer(
        stockfish_path=stockfish_path,
        device=device,
        lr=1e-4,
        batch_size=64
    )

    # Optional resume
    if os.path.exists(
        save_path
    ):
        choice = input(
            "Resume Stockfish model? (y/n): "
        ).lower()

        if choice == "y":
            trainer.load_model(
                save_path
            )

    print(
        "Generating board positions..."
    )

    boards = generate_boards(
        num_positions=500
    )

    print(
        f"Generated {len(boards)} positions"
    )

    trainer.train(
        boards=boards,
        epochs=3,
        depth=12,
        save_path=save_path
    )

    trainer.close()

    print(
        "Stockfish training complete."
    )


if __name__ == "__main__":
    main()