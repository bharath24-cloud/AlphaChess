import os
import chess
import torch
import numpy as np
from tqdm import tqdm

from core.board_encoding import BoardEncoder
from core.move_encoding import MoveEncoder
from core.mcts import MCTS


class SelfPlayGenerator:
    """
    AlphaZero-style self-play data generation
    """

    def __init__(
        self,
        model,
        simulations=100,
        device="cpu",
        temperature=1.0
    ):
        self.model = model
        self.device = device
        self.temperature = temperature

        self.mcts = MCTS(
            model=model,
            simulations=simulations,
            device=device
        )

        self.board_encoder = BoardEncoder()
        self.move_encoder = MoveEncoder()

    def generate_game(
        self,
        max_moves=300
    ):
        """
        Generate one self-play game

        Returns:
            [(board_tensor, policy_target, value)]
        """

        board = chess.Board()

        game_data = []

        move_count = 0

        while (
            not board.is_game_over()
            and move_count < max_moves
        ):

            move, visit_probs = self.search_with_probs(
                board
            )

            board_tensor = self.board_encoder.encode(
                board
            )

            policy_target = self.policy_vector(
                visit_probs
            )

            game_data.append(
                (
                    board_tensor,
                    policy_target,
                    board.turn
                )
            )

            board.push(move)

            move_count += 1

        result_value = self.result_to_value(
            board.result()
        )

        training_data = []

        for (
            board_tensor,
            policy_target,
            player_turn
        ) in game_data:

            value = (
                result_value
                if player_turn == chess.WHITE
                else -result_value
            )

            training_data.append(
                (
                    torch.tensor(
                        board_tensor,
                        dtype=torch.float32
                    ),
                    torch.tensor(
                        policy_target,
                        dtype=torch.float32
                    ),
                    torch.tensor(
                        value,
                        dtype=torch.float32
                    )
                )
            )

        return training_data

    def search_with_probs(
        self,
        board
    ):
        """
        MCTS with visit count probabilities
        """

        root = self.mcts_root(
            board
        )

        for _ in range(
            self.mcts.simulations
        ):

            node = root

            while (
                node.is_expanded()
                and
                len(node.children) > 0
            ):
                _, node = node.select_child(
                    self.mcts.c_puct
                )

            if node.board.is_game_over():

                value = self.mcts.game_result_value(
                    node.board
                )

                node.backpropagate(
                    value
                )

                continue

            move_probs, value = self.mcts.evaluate(
                node.board
            )

            node.expand(
                move_probs
            )

            node.backpropagate(
                value
            )

        visits = np.array(
            [
                child.visit_count
                for child
                in root.children.values()
            ],
            dtype=np.float32
        )

        moves = list(
            root.children.keys()
        )

        if self.temperature == 0:
            best_idx = np.argmax(
                visits
            )

            probs = np.zeros_like(
                visits
            )

            probs[best_idx] = 1

        else:
            visits = visits ** (
                1.0
                /
                self.temperature
            )

            probs = visits / (
                np.sum(visits)
                + 1e-8
            )

        move_idx = np.random.choice(
            len(moves),
            p=probs
        )

        move = moves[
            move_idx
        ]

        visit_probs = {
            move_: prob
            for move_, prob
            in zip(moves, probs)
        }

        return move, visit_probs

    def mcts_root(
        self,
        board
    ):
        """
        Build root node
        """

        from core.mcts import MCTSNode

        root = MCTSNode(
            board
        )

        move_probs, _ = self.mcts.evaluate(
            board
        )

        root.expand(
            move_probs
        )

        return root

    def policy_vector(
        self,
        visit_probs
    ):
        """
        Convert move probabilities
        -> 4672 vector
        """

        vec = np.zeros(
            self.move_encoder.POLICY_SIZE,
            dtype=np.float32
        )

        for move, prob in visit_probs.items():

            idx = self.move_encoder.move_to_index(
                move
            )

            if idx is not None:
                vec[idx] = prob

        return vec

    def result_to_value(
        self,
        result
    ):
        if result == "1-0":
            return 1.0

        if result == "0-1":
            return -1.0

        return 0.0

    def generate_games(
        self,
        num_games=10
    ):
        """
        Generate many games
        """

        all_games = []

        for _ in tqdm(
            range(num_games),
            desc="Self-play"
        ):
            game = self.generate_game()
            all_games.extend(
                game
            )

        return all_games

    def save_games(
        self,
        data,
        path="data/selfplay.pt"
    ):
        os.makedirs(
            os.path.dirname(path),
            exist_ok=True
        )

        torch.save(
            data,
            path
        )

        print(
            f"Saved {len(data)} samples -> {path}"
        )

    def load_games(
        self,
        path="data/selfplay.pt"
    ):
        return torch.load(
            path
        )


if __name__ == "__main__":

    from core.model import AlphaChessNet

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = AlphaChessNet().to(
        device
    )

    generator = SelfPlayGenerator(
        model=model,
        simulations=50,
        device=device,
        temperature=1.0
    )

    data = generator.generate_games(
        num_games=2
    )

    print(
        "Samples:",
        len(data)
    )

    generator.save_games(
        data,
        "data/selfplay.pt"
    )