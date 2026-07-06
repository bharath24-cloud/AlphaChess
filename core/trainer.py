import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging

from core.model import AlphaChessNet
from core.pgn_dataset import (
    PGNDataset,
    StreamingPGNDataset
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlphaChessTrainer:
    def __init__(
        self,
        model=None,
        lr=1e-3,
        batch_size=64,
        device=None
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

        logger.info(
            f"Using device: {self.device}"
        )

        self.model = (
            model
            if model
            else AlphaChessNet()
        )

        self.model.to(
            self.device
        )

        self.batch_size = batch_size

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=1e-4
        )

        self.policy_loss_fn = (
            nn.CrossEntropyLoss()
        )

        self.value_loss_fn = (
            nn.MSELoss()
        )

    def train_from_pgn(
        self,
        pgn_path,
        epochs=5,
        max_games=None,
        streaming=True,
        save_path="saved_models/alphachess_v1.pth"
    ):
        """
        Train from Lichess PGN
        """

        logger.info(
            "Loading dataset..."
        )

        if streaming:
            dataset = StreamingPGNDataset(
                pgn_path,
                max_games=max_games
            )
        else:
            dataset = PGNDataset(
                pgn_path,
                max_games=max_games
            )

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )

        logger.info(
            f"Dataset size: {len(dataset)}"
        )

        for epoch in range(
            epochs
        ):
            self.model.train()

            total_loss = 0
            total_policy = 0
            total_value = 0

            progress = tqdm(
                loader,
                desc=f"Epoch {epoch+1}/{epochs}"
            )

            for (
                boards,
                policy_targets,
                value_targets
            ) in progress:

                boards = boards.to(
                    self.device
                )

                policy_targets = (
                    policy_targets.to(
                        self.device
                    )
                )

                value_targets = (
                    value_targets.to(
                        self.device
                    )
                    .unsqueeze(1)
                )

                self.optimizer.zero_grad()

                policy_out, value_out = (
                    self.model(
                        boards
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

                total_policy += (
                    policy_loss.item()
                )

                total_value += (
                    value_loss.item()
                )

                progress.set_postfix(
                    loss=f"{loss.item():.4f}",
                    policy=f"{policy_loss.item():.4f}",
                    value=f"{value_loss.item():.4f}"
                )

            avg_loss = (
                total_loss
                /
                len(loader)
            )

            avg_policy = (
                total_policy
                /
                len(loader)
            )

            avg_value = (
                total_value
                /
                len(loader)
            )

            logger.info(
                f"Epoch {epoch+1} "
                f"Loss={avg_loss:.4f} "
                f"Policy={avg_policy:.4f} "
                f"Value={avg_value:.4f}"
            )

            self.save_model(
                save_path
            )

        logger.info(
            "Training complete"
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

        logger.info(
            f"Model saved: {path}"
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

        logger.info(
            f"Loaded model: {path}"
        )


if __name__ == "__main__":

    trainer = AlphaChessTrainer(
        batch_size=64
    )

    trainer.train_from_pgn(
        pgn_path="data/lichess_games.pgn",
        epochs=5,
        max_games=15000,
        streaming=True,
        save_path="saved_models/alphachess_v1.pth"
    )