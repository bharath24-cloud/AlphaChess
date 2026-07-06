import chess


class MoveEncoder:
    """
    Move <-> Policy index encoder

    Policy size = 4672
    Based on:
        from_square * 73 + move_type

    64 squares * 73 = 4672
    """

    POLICY_SIZE = 4672

    def __init__(self):
        pass

    def move_to_index(self, move: chess.Move):
        """
        Convert chess.Move -> policy index
        """

        from_sq = move.from_square
        to_sq = move.to_square

        from_rank = chess.square_rank(from_sq)
        from_file = chess.square_file(from_sq)

        to_rank = chess.square_rank(to_sq)
        to_file = chess.square_file(to_sq)

        dr = to_rank - from_rank
        df = to_file - from_file

        move_type = self._direction_index(
            dr,
            df,
            move.promotion
        )

        if move_type is None:
            return None

        index = from_sq * 73 + move_type

        if index >= self.POLICY_SIZE:
            return None

        return index

    def index_to_move(self, board, index):
        """
        Convert policy index -> legal move
        """

        from_sq = index // 73
        move_type = index % 73

        candidate = self._decode_move(
            from_sq,
            move_type
        )

        if candidate in board.legal_moves:
            return candidate

        # fallback search
        for move in board.legal_moves:
            idx = self.move_to_index(move)
            if idx == index:
                return move

        return None

    def legal_moves_mask(self, board):
        """
        Mask for legal moves
        Shape: (4672,)
        """

        mask = [0] * self.POLICY_SIZE

        for move in board.legal_moves:
            idx = self.move_to_index(move)

            if idx is not None:
                mask[idx] = 1

        return mask

    def _direction_index(
        self,
        dr,
        df,
        promotion=None
    ):
        """
        AlphaZero-style 73 move planes

        0-55  Queen-like moves
        56-63 Knight moves
        64-72 Promotions
        """

        directions = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        ]

        # Queen-like moves
        for d_idx, (rdir, fdir) in enumerate(directions):
            for dist in range(1, 8):
                if (
                    dr == rdir * dist
                    and
                    df == fdir * dist
                ):
                    return d_idx * 7 + (dist - 1)

        # Knight moves
        knight_moves = [
            (2, 1),
            (1, 2),
            (-1, 2),
            (-2, 1),
            (-2, -1),
            (-1, -2),
            (1, -2),
            (2, -1),
        ]

        for k_idx, (kr, kf) in enumerate(knight_moves):
            if dr == kr and df == kf:
                return 56 + k_idx

        # Promotions
        if promotion is not None:
            promo_map = {
                chess.KNIGHT: 64,
                chess.BISHOP: 67,
                chess.ROOK: 70,
            }

            if promotion == chess.QUEEN:
                return 72

            return promo_map.get(
                promotion,
                None
            )

        return None

    def _decode_move(
        self,
        from_sq,
        move_type
    ):
        """
        Decode policy move
        """

        rank = chess.square_rank(from_sq)
        file = chess.square_file(from_sq)

        directions = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        ]

        # Queen-like
        if move_type < 56:
            direction = move_type // 7
            distance = (move_type % 7) + 1

            dr, df = directions[direction]

            r = rank + dr * distance
            f = file + df * distance

            if 0 <= r < 8 and 0 <= f < 8:
                to_sq = chess.square(
                    f,
                    r
                )
                return chess.Move(
                    from_sq,
                    to_sq
                )

        # Knight
        if 56 <= move_type < 64:
            knight_moves = [
                (2, 1),
                (1, 2),
                (-1, 2),
                (-2, 1),
                (-2, -1),
                (-1, -2),
                (1, -2),
                (2, -1),
            ]

            dr, df = knight_moves[
                move_type - 56
            ]

            r = rank + dr
            f = file + df

            if 0 <= r < 8 and 0 <= f < 8:
                to_sq = chess.square(
                    f,
                    r
                )
                return chess.Move(
                    from_sq,
                    to_sq
                )

        return chess.Move.null()


if __name__ == "__main__":
    encoder = MoveEncoder()

    board = chess.Board()

    move = chess.Move.from_uci(
        "e2e4"
    )

    idx = encoder.move_to_index(
        move
    )

    print(
        "Move:",
        move
    )

    print(
        "Index:",
        idx
    )

    recovered = encoder.index_to_move(
        board,
        idx
    )

    print(
        "Recovered:",
        recovered
    )