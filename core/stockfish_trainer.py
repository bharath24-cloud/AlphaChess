import os
import chess
import chess.engine
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np

from core.board_encoding import BoardEncoder
from core.move_encoding import MoveEncoder
from core.model import AlphaChessNet


class StockfishTrainer:
    """
    Stockfish-guided training

    Uses:
        Board -> Stockfish best move
        Board -> Stockfish evaluation
    """

    def __init__(
        self,
        stockfish_path,
        model=None,
        device=None,
        lr=1e-4,
        batch_size=64
    ):
        self.device = (
            device
            if device
            else (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        )

        self.model = (
            model
            if model
            else AlphaChessNet()
        ).to(self.device)

        self.batch_size = batch_size

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=1e-4
        )

        self.policy_loss_fn = nn.CrossEntropyLoss()
        self.value_loss_fn = nn.MSELoss()

        self.board_encoder = BoardEncoder()
        self.move_encoder = MoveEncoder()

        self.engine = chess.engine.SimpleEngine.popen_uci(
            stockfish_path
        )

    def evaluate_position(
        self,
        board,
        depth=12
    ):
        """
        Get Stockfish best move + evaluation
        """

        result = self.engine.play(
            board,
            chess.engine.Limit(
                depth=depth
            )
        )

        info = self.engine.analyse(
            board,
            chess.engine.Limit(
                depth=depth
            )
        )

        score = info["score"].white()

        if score.is_mate():
            mate = score.mate()

            if mate > 0:
                value = 1.0
            else:
                value = -1.0
        else:
            cp = score.score()

            value = np.tanh(
                cp / 500.0
            )

        best_move = result.move

        return best_move, float(value)

    def build_dataset(
        self,
        boards,
        depth=12
    ):
        """
        Create Stockfish-labelled dataset
        """

        board_tensors = []
        policy_targets = []
        value_targets = []

        for board in tqdm(
            boards,
            desc="Stockfish analysis"
        ):

            best_move, value = (
                self.evaluate_position(
                    board,
                    depth
                )
            )

            move_idx = (
                self.move_encoder.move_to_index(
                    best_move
                )
            )

            if move_idx is None:
                continue

            board_tensor = (
                self.board_encoder.encode(
                    board
                )
            )

            board_tensors.append(
                board_tensor
            )

            policy_targets.append(
                move_idx
            )

            value_targets.append(
                value
            )

        return TensorDataset(
            torch.tensor(
                board_tensors,
                dtype=torch.float32
            ),
            torch.tensor(
                policy_targets,
                dtype=torch.long
            ),
            torch.tensor(
                value_targets,
                dtype=torch.float32
            )
        )

    def train(
        self,
        boards,
        epochs=3,
        depth=12,
        save_path="saved_models/alphachess_stockfish.pth"
    ):
        """
        Train using Stockfish guidance
        """

        dataset = self.build_dataset(
            boards,
            depth
        )

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        for epoch in range(
            epochs
        ):
            self.model.train()

            total_loss = 0

            progress = tqdm(
                loader,
                desc=f"Epoch {epoch+1}/{epochs}"
            )

            for (
                boards_batch,
                policy_targets,
                value_targets
            ) in progress:

                boards_batch = (
                    boards_batch.to(
                        self.device
                    )
                )

                policy_targets = (
                    policy_targets.to(
                        self.device
                    )
                )

                value_targets = (
                    value_targets
                    .unsqueeze(1)
                    .to(self.device)
                )

                self.optimizer.zero_grad()

                policy_out, value_out = (
                    self.model(
                        boards_batch
                    )
                )

                policy_loss = (
                    self.policy_loss_fn(
                        policy_out,
                        policy_targets
                    )
                )

                value_loss = (
                    self.value_loss_fn(
                        value_out,
                        value_targets
                    )
                )

                loss = (
                    policy_loss
                    +
                    value_loss
                )

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    1.0
                )

                self.optimizer.step()

                total_loss += (
                    loss.item()
                )

                progress.set_postfix(
                    loss=f"{loss.item():.4f}"
                )

            avg_loss = (
                total_loss
                /
                len(loader)
            )

            print(
                f"Epoch {epoch+1} "
                f"Average Loss: {avg_loss:.4f}"
            )

            self.save_model(
                save_path
            )

    def save_model(
        self,
        path
    ):
        os.makedirs(
            os.path.dirname(path),
            exist_ok=True
        )

        torch.save(
            {
                "model_state_dict":
                    self.model.state_dict(),
                "optimizer_state_dict":
                    self.optimizer.state_dict(),
            },
            path
        )

        print(
            f"Saved -> {path}"
        )

    def load_model(
        self,
        path
    ):
        checkpoint = torch.load(
            path,
            map_location=self.device
        )

        self.model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        if (
            "optimizer_state_dict"
            in checkpoint
        ):
            self.optimizer.load_state_dict(
                checkpoint[
                    "optimizer_state_dict"
                ]
            )

        self.model.to(
            self.device
        )

        print(
            f"Loaded -> {path}"
        )

    def close(self):
        self.engine.quit()


if __name__ == "__main__":

    stockfish_path = (
        "stockfish/stockfish.exe"
    )

    trainer = StockfishTrainer(
        stockfish_path
    )

    boards = []

    board = chess.Board()

    boards.append(
        board.copy()
    )

    board.push_san(
        "e4"
    )
    boards.append(
        board.copy()
    )

    board.push_san(
        "e5"
    )
    boards.append(
        board.copy()
    )

    trainer.train(
        boards,
        epochs=2,
        depth=10
    )

    trainer.close()