import os
import sys
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

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
from core.self_play import SelfPlayGenerator


class SelfPlayTrainer:
    def __init__(
        self,
        model_path=None,
        lr=1e-4,
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

        print(
            f"Using device: {self.device}"
        )

        self.model = AlphaChessNet().to(
            self.device
        )

        if (
            model_path
            and
            os.path.exists(model_path)
        ):
            self.load_model(
                model_path
            )

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

        self.batch_size = batch_size

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

        print(
            f"Loaded: {path}"
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
                    self.optimizer.state_dict()
            },
            path
        )

        print(
            f"Saved: {path}"
        )

    def train_on_selfplay(
        self,
        num_games=20,
        simulations=100,
        epochs=3,
        temperature=1.0,
        save_path="saved_models/alphachess_selfplay.pth"
    ):
        """
        Generate self-play games
        and train on them
        """

        generator = SelfPlayGenerator(
            model=self.model,
            simulations=simulations,
            device=self.device,
            temperature=temperature
        )

        print(
            "Generating self-play games..."
        )

        data = generator.generate_games(
            num_games=num_games
        )

        boards = []
        policy_targets = []
        value_targets = []

        for (
            board,
            policy,
            value
        ) in data:

            boards.append(
                board
            )

            policy_targets.append(
                torch.argmax(
                    policy
                )
            )

            value_targets.append(
                value
            )

        dataset = TensorDataset(
            torch.stack(
                boards
            ),
            torch.tensor(
                policy_targets,
                dtype=torch.long
            ),
            torch.stack(
                value_targets
            )
        )

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        print(
            f"Training on {len(dataset)} samples"
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
                policy_batch,
                value_batch
            ) in progress:

                boards_batch = (
                    boards_batch.to(
                        self.device
                    )
                )

                policy_batch = (
                    policy_batch.to(
                        self.device
                    )
                )

                value_batch = (
                    value_batch
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
                        policy_batch
                    )
                )

                value_loss = (
                    self.value_loss_fn(
                        value_out,
                        value_batch
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
                f"Average Loss: "
                f"{avg_loss:.4f}"
            )

            self.save_model(
                save_path
            )

        print(
            "Self-play training complete."
        )


def main():

    model_path = (
        "saved_models/alphachess_v1.pth"
    )

    trainer = SelfPlayTrainer(
        model_path=model_path,
        lr=1e-4,
        batch_size=64
    )

    trainer.train_on_selfplay(
        num_games=20,
        simulations=100,
        epochs=3,
        temperature=1.0,
        save_path="saved_models/alphachess_selfplay.pth"
    )


if __name__ == "__main__":
    main()