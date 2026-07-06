import math
import numpy as np
import chess
import torch

from core.move_encoding import MoveEncoder
from core.board_encoding import BoardEncoder


class MCTSNode:
    def __init__(
        self,
        board,
        parent=None,
        prior=0.0
    ):
        self.board = board.copy()
        self.parent = parent
        self.prior = prior

        self.children = {}

        self.visit_count = 0
        self.value_sum = 0.0

    @property
    def value(self):
        if self.visit_count == 0:
            return 0.0

        return self.value_sum / self.visit_count

    def is_expanded(self):
        return len(self.children) > 0

    def expand(
        self,
        policy_probs
    ):
        """
        Expand node using policy probabilities
        """

        for move, prob in policy_probs.items():

            next_board = self.board.copy()
            next_board.push(move)

            self.children[move] = MCTSNode(
                next_board,
                parent=self,
                prior=prob
            )

    def select_child(
        self,
        c_puct=1.5
    ):
        """
        Select child using PUCT
        """

        best_score = -float("inf")
        best_move = None
        best_child = None

        for move, child in self.children.items():

            score = self._puct_score(
                child,
                c_puct
            )

            if score > best_score:
                best_score = score
                best_move = move
                best_child = child

        return best_move, best_child

    def _puct_score(
        self,
        child,
        c_puct
    ):
        """
        PUCT formula
        """

        q_value = -child.value

        u_value = (
            c_puct
            *
            child.prior
            *
            math.sqrt(
                self.visit_count + 1
            )
            /
            (
                1
                +
                child.visit_count
            )
        )

        return q_value + u_value

    def backpropagate(
        self,
        value
    ):
        """
        Update node statistics
        """

        self.visit_count += 1
        self.value_sum += value

        if self.parent:
            self.parent.backpropagate(
                -value
            )


class MCTS:
    def __init__(
        self,
        model,
        simulations=100,
        c_puct=1.5,
        device="cpu"
    ):
        self.model = model
        self.simulations = simulations
        self.c_puct = c_puct
        self.device = device

        self.move_encoder = MoveEncoder()
        self.board_encoder = BoardEncoder()

        self.model.eval()

    def search(
        self,
        board
    ):
        """
        Run MCTS search
        """

        root = MCTSNode(
            board
        )

        policy_probs, _ = self.evaluate(
            board
        )

        root.expand(
            policy_probs
        )

        for _ in range(
            self.simulations
        ):
            node = root

            # Selection
            while (
                node.is_expanded()
                and
                len(node.children) > 0
            ):
                _, node = node.select_child(
                    self.c_puct
                )

            # Terminal node
            if node.board.is_game_over():

                result = self.game_result_value(
                    node.board
                )

                node.backpropagate(
                    result
                )

                continue

            # Expansion + Evaluation
            policy_probs, value = self.evaluate(
                node.board
            )

            node.expand(
                policy_probs
            )

            node.backpropagate(
                value
            )

        return self.select_move(
            root
        )

    def evaluate(
        self,
        board
    ):
        """
        Neural network evaluation
        """

        board_tensor = self.board_encoder.encode(
            board
        )

        board_tensor = torch.tensor(
            board_tensor,
            dtype=torch.float32
        ).unsqueeze(0).to(
            self.device
        )

        with torch.no_grad():
            policy_logits, value = (
                self.model(
                    board_tensor
                )
            )

        policy_logits = (
            policy_logits
            .cpu()
            .numpy()[0]
        )

        value = float(
            value.item()
        )

        legal_moves = list(
            board.legal_moves
        )

        move_probs = {}

        logits = []

        for move in legal_moves:
            idx = self.move_encoder.move_to_index(
                move
            )

            if idx is not None:
                logits.append(
                    policy_logits[idx]
                )
            else:
                logits.append(
                    -1e9
                )

        probs = self.softmax(
            np.array(
                logits
            )
        )

        for move, prob in zip(
            legal_moves,
            probs
        ):
            move_probs[
                move
            ] = float(prob)

        return move_probs, value

    def select_move(
        self,
        root
    ):
        """
        Choose move with highest visit count
        """

        best_move = max(
            root.children.items(),
            key=lambda item:
                item[1].visit_count
        )[0]

        return best_move

    def softmax(
        self,
        x
    ):
        x = x - np.max(
            x
        )

        exp_x = np.exp(
            x
        )

        return exp_x / (
            np.sum(exp_x)
            + 1e-8
        )

    def game_result_value(
        self,
        board
    ):
        """
        Terminal result value
        """

        result = board.result()

        if result == "1-0":
            return 1.0

        if result == "0-1":
            return -1.0

        return 0.0


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

    mcts = MCTS(
        model=model,
        simulations=50,
        device=device
    )

    board = chess.Board()

    move = mcts.search(
        board
    )

    print(
        "Best move:",
        move
    )