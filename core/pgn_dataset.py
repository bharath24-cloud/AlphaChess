import chess
import chess.pgn
import torch
from torch.utils.data import Dataset
import numpy as np
import logging
from tqdm import tqdm

from core.board_encoding import BoardEncoder
from core.move_encoding import MoveEncoder


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PGNDataset(Dataset):
    """
    Streaming Lichess PGN Dataset

    Returns:
        board_tensor : (19,8,8)
        policy_target : int
        value_target : float
    """

    def __init__(
        self,
        pgn_path,
        max_games=None,
        min_elo=0
    ):
        self.pgn_path = pgn_path
        self.max_games = max_games
        self.min_elo = min_elo

        self.board_encoder = BoardEncoder()
        self.move_encoder = MoveEncoder()

        self.samples = []

        logger.info(
            "Building dataset from PGN..."
        )

        self._load_games()

        logger.info(
            f"Dataset ready with {len(self.samples)} samples"
        )

    def _load_games(self):
        """
        Parse PGN and store lightweight samples:
            (fen, move_index, result)
        """

        game_count = 0

        with open(
            self.pgn_path,
            encoding="utf-8",
            errors="ignore"
        ) as pgn:

            while True:
                game = chess.pgn.read_game(pgn)

                if game is None:
                    break

                if (
                    self.max_games is not None
                    and
                    game_count >= self.max_games
                ):
                    break

                white_elo = int(
                    game.headers.get(
                        "WhiteElo",
                        0
                    )
                )

                black_elo = int(
                    game.headers.get(
                        "BlackElo",
                        0
                    )
                )

                if (
                    white_elo < self.min_elo
                    or
                    black_elo < self.min_elo
                ):
                    continue

                result = game.headers.get(
                    "Result",
                    "*"
                )

                value = self._result_to_value(
                    result
                )

                board = game.board()

                for move in game.mainline_moves():

                    move_idx = self.move_encoder.move_to_index(
                        move
                    )

                    if move_idx is not None:
                        self.samples.append(
                            (
                                board.fen(),
                                move_idx,
                                value
                            )
                        )

                    board.push(move)

                game_count += 1

                if game_count % 1000 == 0:
                    logger.info(
                        f"Processed {game_count} games, "
                        f"{len(self.samples)} samples"
                    )

    def _result_to_value(
        self,
        result
    ):
        """
        Convert PGN result to value target
        """

        if result == "1-0":
            return 1.0

        if result == "0-1":
            return -1.0

        if result == "1/2-1/2":
            return 0.0

        return 0.0

    def __len__(self):
        return len(
            self.samples
        )

    def __getitem__(
        self,
        idx
    ):
        fen, move_idx, value = self.samples[
            idx
        ]

        board = chess.Board(
            fen
        )

        board_tensor = self.board_encoder.encode(
            board
        )

        board_tensor = torch.tensor(
            board_tensor,
            dtype=torch.float32
        )

        policy_target = torch.tensor(
            move_idx,
            dtype=torch.long
        )

        value_target = torch.tensor(
            value,
            dtype=torch.float32
        )

        return (
            board_tensor,
            policy_target,
            value_target
        )


class StreamingPGNDataset(
    Dataset
):
    """
    RAM-safe streaming dataset
    Loads samples lazily from PGN

    Better for huge Lichess files
    """

    def __init__(
        self,
        pgn_path,
        max_games=None
    ):
        self.pgn_path = pgn_path
        self.max_games = max_games

        self.board_encoder = BoardEncoder()
        self.move_encoder = MoveEncoder()

        self.samples = []

        logger.info(
            "Indexing PGN..."
        )

        self._index_games()

    def _index_games(self):
        game_count = 0

        with open(
            self.pgn_path,
            encoding="utf-8",
            errors="ignore"
        ) as pgn:

            while True:
                offset = pgn.tell()

                game = chess.pgn.read_game(
                    pgn
                )

                if game is None:
                    break

                if (
                    self.max_games
                    and
                    game_count >= self.max_games
                ):
                    break

                self.samples.append(
                    offset
                )

                game_count += 1

        logger.info(
            f"Indexed {len(self.samples)} games"
        )

    def __len__(self):
        return len(
            self.samples
        )

    def __getitem__(
        self,
        idx
    ):
        offset = self.samples[
            idx
        ]

        with open(
            self.pgn_path,
            encoding="utf-8",
            errors="ignore"
        ) as pgn:

            pgn.seek(
                offset
            )

            game = chess.pgn.read_game(
                pgn
            )

            board = game.board()

            moves = list(
                game.mainline_moves()
            )

            if len(moves) == 0:
                return self.__getitem__(
                    (idx + 1)
                    %
                    len(self.samples)
                )

            move = moves[
                np.random.randint(
                    len(moves)
                )
            ]

            result = game.headers.get(
                "Result",
                "*"
            )

            value = 0.0

            if result == "1-0":
                value = 1.0
            elif result == "0-1":
                value = -1.0

            for m in moves:
                if m == move:
                    break
                board.push(m)

            board_tensor = torch.tensor(
                self.board_encoder.encode(
                    board
                ),
                dtype=torch.float32
            )

            move_idx = self.move_encoder.move_to_index(
                move
            )

            policy_target = torch.tensor(
                move_idx,
                dtype=torch.long
            )

            value_target = torch.tensor(
                value,
                dtype=torch.float32
            )

            return (
                board_tensor,
                policy_target,
                value_target
            )


if __name__ == "__main__":

    dataset = PGNDataset(
        "data/lichess_games.pgn",
        max_games=100
    )

    print(
        "Total samples:",
        len(dataset)
    )

    board, policy, value = dataset[
        0
    ]

    print(
        "Board shape:",
        board.shape
    )

    print(
        "Policy:",
        policy
    )

    print(
        "Value:",
        value
    )