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

from core.model import AlphaChessNet
from core.mcts import MCTS


class ChessAIPlayer:
    def __init__(
        self,
        model_path,
        simulations=100
    ):
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            f"Using device: {self.device}"
        )

        self.model = AlphaChessNet().to(
            self.device
        )

        self.load_model(
            model_path
        )

        self.mcts = MCTS(
            model=self.model,
            simulations=simulations,
            device=self.device
        )

    def load_model(
        self,
        model_path
    ):
        checkpoint = torch.load(
            model_path,
            map_location=self.device
        )

        self.model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        self.model.eval()

        print(
            f"Loaded model: {model_path}"
        )

    def choose_move(
        self,
        board
    ):
        return self.mcts.search(
            board
        )


def print_board(
    board
):
    print("\n")
    print(board)
    print("\n")

    print(
        "FEN:",
        board.fen()
    )

    print(
        "Turn:",
        "White"
        if board.turn
        else "Black"
    )
    print()


def human_move(
    board
):
    while True:

        user_input = input(
            "Your move (e2e4): "
        ).strip()

        try:
            move = chess.Move.from_uci(
                user_input
            )

            if move in board.legal_moves:
                return move

            print(
                "Illegal move."
            )

        except:
            print(
                "Invalid format."
            )


def play_game():

    model_path = (
        "saved_models/alphachess_v1.pth"
    )

    ai = ChessAIPlayer(
        model_path=model_path,
        simulations=100
    )

    board = chess.Board()

    print(
        "\nAlphaChess vs Human\n"
    )

    choice = input(
        "Play as (w/b): "
    ).lower()

    human_is_white = (
        choice == "w"
    )

    while not board.is_game_over():

        print_board(
            board
        )

        if (
            board.turn == chess.WHITE
            and human_is_white
        ) or (
            board.turn == chess.BLACK
            and not human_is_white
        ):

            move = human_move(
                board
            )

            board.push(
                move
            )

        else:

            print(
                "AI thinking..."
            )

            ai_move = ai.choose_move(
                board
            )

            print(
                f"AI move: {ai_move}"
            )

            board.push(
                ai_move
            )

    print_board(
        board
    )

    print(
        "Game Over"
    )

    print(
        "Result:",
        board.result()
    )


if __name__ == "__main__":
    play_game()