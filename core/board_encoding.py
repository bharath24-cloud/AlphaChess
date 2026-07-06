import chess
import numpy as np


class BoardEncoder:
    """
    AlphaZero-style 19-channel board encoder

    Channels:
    0-5   : White pieces
    6-11  : Black pieces
    12    : Side to move
    13    : White kingside castling
    14    : White queenside castling
    15    : Black kingside castling
    16    : Black queenside castling
    17    : En passant square
    18    : Halfmove clock
    """

    PIECE_TO_CHANNEL = {
        (chess.PAWN, chess.WHITE): 0,
        (chess.KNIGHT, chess.WHITE): 1,
        (chess.BISHOP, chess.WHITE): 2,
        (chess.ROOK, chess.WHITE): 3,
        (chess.QUEEN, chess.WHITE): 4,
        (chess.KING, chess.WHITE): 5,

        (chess.PAWN, chess.BLACK): 6,
        (chess.KNIGHT, chess.BLACK): 7,
        (chess.BISHOP, chess.BLACK): 8,
        (chess.ROOK, chess.BLACK): 9,
        (chess.QUEEN, chess.BLACK): 10,
        (chess.KING, chess.BLACK): 11,
    }

    def __init__(self):
        self.channels = 19

    def encode(self, board: chess.Board):
        """
        Returns:
            numpy array (19, 8, 8)
        """
        tensor = np.zeros(
            (19, 8, 8),
            dtype=np.float32
        )

        # Piece planes
        for square in chess.SQUARES:
            piece = board.piece_at(square)

            if piece is not None:
                channel = self.PIECE_TO_CHANNEL[
                    (piece.piece_type, piece.color)
                ]

                row = 7 - chess.square_rank(square)
                col = chess.square_file(square)

                tensor[channel][row][col] = 1.0

        # Side to move
        if board.turn == chess.WHITE:
            tensor[12][:][:] = 1.0

        # Castling rights
        if board.has_kingside_castling_rights(chess.WHITE):
            tensor[13][:][:] = 1.0

        if board.has_queenside_castling_rights(chess.WHITE):
            tensor[14][:][:] = 1.0

        if board.has_kingside_castling_rights(chess.BLACK):
            tensor[15][:][:] = 1.0

        if board.has_queenside_castling_rights(chess.BLACK):
            tensor[16][:][:] = 1.0

        # En passant
        if board.ep_square is not None:
            row = 7 - chess.square_rank(board.ep_square)
            col = chess.square_file(board.ep_square)

            tensor[17][row][col] = 1.0

        # Halfmove clock
        tensor[18][:][:] = min(
            board.halfmove_clock / 100.0,
            1.0
        )

        return tensor

    def encode_batch(self, boards):
        """
        Encode list of boards

        Returns:
            numpy array (batch, 19, 8, 8)
        """
        encoded = [
            self.encode(board)
            for board in boards
        ]

        return np.array(
            encoded,
            dtype=np.float32
        )


if __name__ == "__main__":
    encoder = BoardEncoder()

    board = chess.Board()

    tensor = encoder.encode(board)

    print("Shape:", tensor.shape)
    print("Channels:", tensor.shape[0])
    print("Board tensor created successfully")