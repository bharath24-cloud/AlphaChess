import os
import sys
import chess
import torch
import pygame

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


WIDTH = 640
HEIGHT = 640
SQ_SIZE = WIDTH // 8


PIECE_FILES = {
    'P': 'wp.png',
    'N': 'wn.png',
    'B': 'wb.png',
    'R': 'wr.png',
    'Q': 'wq.png',
    'K': 'wk.png',
    'p': 'bp.png',
    'n': 'bn.png',
    'b': 'bb.png',
    'r': 'br.png',
    'q': 'bq.png',
    'k': 'bk.png',
}


LIGHT = (240, 217, 181)
DARK = (181, 136, 99)
HIGHLIGHT = (186, 202, 68)


class ChessGUI:
    def __init__(
        self,
        model_path,
        simulations=100
    ):
        pygame.init()

        self.screen = pygame.display.set_mode(
            (WIDTH, HEIGHT)
        )

        pygame.display.set_caption(
            "AlphaChess"
        )

        self.board = chess.Board()

        self.selected_square = None

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            f"Device: {self.device}"
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

        self.piece_images = {}
        self.load_images()

        self.human_is_white = True

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

        self.model.eval()

        print(
            f"Loaded model: {path}"
        )

    def load_images(
        self
    ):
        base = os.path.join(
            os.path.dirname(__file__),
            "pieces"
        )

        for piece, file in PIECE_FILES.items():

            path = os.path.join(
                base,
                file
            )

            if not os.path.exists(path):
                print(
                    f"Missing image: {path}"
                )
                continue

            image = pygame.image.load(
                path
            )

            image = pygame.transform.scale(
                image,
                (SQ_SIZE, SQ_SIZE)
            )

            self.piece_images[
                piece
            ] = image

    def draw_board(
        self
    ):
        for row in range(8):
            for col in range(8):

                color = (
                    LIGHT
                    if (row + col) % 2 == 0
                    else DARK
                )

                pygame.draw.rect(
                    self.screen,
                    color,
                    pygame.Rect(
                        col * SQ_SIZE,
                        row * SQ_SIZE,
                        SQ_SIZE,
                        SQ_SIZE
                    )
                )

        if self.selected_square is not None:

            row = (
                7
                -
                chess.square_rank(
                    self.selected_square
                )
            )

            col = chess.square_file(
                self.selected_square
            )

            pygame.draw.rect(
                self.screen,
                HIGHLIGHT,
                pygame.Rect(
                    col * SQ_SIZE,
                    row * SQ_SIZE,
                    SQ_SIZE,
                    SQ_SIZE
                ),
                4
            )

    def draw_pieces(
        self
    ):
        for square in chess.SQUARES:

            piece = self.board.piece_at(
                square
            )

            if piece is None:
                continue

            symbol = piece.symbol()

            image = self.piece_images.get(
                symbol
            )

            if image is None:
                continue

            row = (
                7
                -
                chess.square_rank(
                    square
                )
            )

            col = chess.square_file(
                square
            )

            self.screen.blit(
                image,
                (
                    col * SQ_SIZE,
                    row * SQ_SIZE
                )
            )

    def mouse_to_square(
        self,
        pos
    ):
        x, y = pos

        col = x // SQ_SIZE
        row = y // SQ_SIZE

        board_row = 7 - row

        return chess.square(
            col,
            board_row
        )

    def handle_click(
        self,
        pos
    ):
        if self.board.turn != (
            chess.WHITE
            if self.human_is_white
            else chess.BLACK
        ):
            return

        square = self.mouse_to_square(
            pos
        )

        if self.selected_square is None:

            piece = self.board.piece_at(
                square
            )

            if (
                piece
                and
                piece.color
                ==
                self.board.turn
            ):
                self.selected_square = square

        else:

            move = chess.Move(
                self.selected_square,
                square
            )

            # Promotion
            piece = self.board.piece_at(
                self.selected_square
            )

            if (
                piece
                and
                piece.piece_type
                ==
                chess.PAWN
            ):
                rank = chess.square_rank(
                    square
                )

                if rank == 0 or rank == 7:
                    move = chess.Move(
                        self.selected_square,
                        square,
                        promotion=chess.QUEEN
                    )

            if move in self.board.legal_moves:

                self.board.push(
                    move
                )

                self.selected_square = None

                if not self.board.is_game_over():
                    self.ai_move()

            else:
                self.selected_square = None

    def ai_move(
        self
    ):
        print(
            "AI thinking..."
        )

        move = self.mcts.search(
            self.board
        )

        print(
            f"AI move: {move}"
        )

        self.board.push(
            move
        )

    def run(
        self
    ):
        running = True
        clock = pygame.time.Clock()

        while running:

            for event in pygame.event.get():

                if (
                    event.type
                    ==
                    pygame.QUIT
                ):
                    running = False

                elif (
                    event.type
                    ==
                    pygame.MOUSEBUTTONDOWN
                ):
                    self.handle_click(
                        event.pos
                    )

            self.draw_board()
            self.draw_pieces()

            pygame.display.flip()
            clock.tick(60)

            if self.board.is_game_over():

                print(
                    "Game Over:",
                    self.board.result()
                )

        pygame.quit()


if __name__ == "__main__":

    gui = ChessGUI(
        model_path="saved_models/alphachess_v1.pth",
        simulations=200
    )

    gui.run()