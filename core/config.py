import os
import torch


class Config:

    # =========================
    # Project Paths
    # =========================

    ROOT_DIR = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )

    DATA_DIR = os.path.join(
        ROOT_DIR,
        "data"
    )

    MODEL_DIR = os.path.join(
        ROOT_DIR,
        "saved_models"
    )

    LOG_DIR = os.path.join(
        ROOT_DIR,
        "logs"
    )

    STOCKFISH_DIR = os.path.join(
        ROOT_DIR,
        "stockfish"
    )

    GUI_PIECES_DIR = os.path.join(
        ROOT_DIR,
        "gui",
        "pieces"
    )

    # =========================
    # Device
    # =========================

    DEVICE = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # =========================
    # Model
    # =========================

    INPUT_CHANNELS = 19
    CHANNELS = 256
    RESIDUAL_BLOCKS = 10
    POLICY_SIZE = 4672

    # =========================
    # Supervised Training
    # =========================

    SUPERVISED_EPOCHS = 5
    SUPERVISED_BATCH_SIZE = 64
    SUPERVISED_LR = 1e-3
    MAX_GAMES = 15000
    STREAMING_DATASET = True

    # =========================
    # Self Play
    # =========================

    SELFPLAY_GAMES = 20
    SELFPLAY_SIMULATIONS = 100
    SELFPLAY_EPOCHS = 3
    SELFPLAY_LR = 1e-4
    TEMPERATURE = 1.0

    # =========================
    # MCTS
    # =========================

    MCTS_SIMULATIONS = 100
    C_PUCT = 1.5

    # =========================
    # Stockfish Training
    # =========================

    STOCKFISH_DEPTH = 12
    STOCKFISH_BATCH_SIZE = 64
    STOCKFISH_EPOCHS = 3
    STOCKFISH_LR = 1e-4

    # Update executable name if needed
    STOCKFISH_PATH = os.path.join(
        STOCKFISH_DIR,
        "stockfish.exe"
    )

    # =========================
    # Saved Models
    # =========================

    SUPERVISED_MODEL = os.path.join(
        MODEL_DIR,
        "alphachess_v1.pth"
    )

    SELFPLAY_MODEL = os.path.join(
        MODEL_DIR,
        "alphachess_selfplay.pth"
    )

    STOCKFISH_MODEL = os.path.join(
        MODEL_DIR,
        "alphachess_stockfish.pth"
    )

    # =========================
    # GUI
    # =========================

    BOARD_SIZE = 640
    FPS = 60

    # =========================
    # Utility
    # =========================

    @staticmethod
    def create_dirs():
        """
        Create required folders
        """

        dirs = [
            Config.DATA_DIR,
            Config.MODEL_DIR,
            Config.LOG_DIR,
            Config.STOCKFISH_DIR,
            Config.GUI_PIECES_DIR
        ]

        for d in dirs:
            os.makedirs(
                d,
                exist_ok=True
            )


if __name__ == "__main__":

    Config.create_dirs()

    print(
        "AlphaChess configuration loaded"
    )

    print(
        "Device:",
        Config.DEVICE
    )

    print(
        "Model dir:",
        Config.MODEL_DIR
    )

    print(
        "Stockfish:",
        Config.STOCKFISH_PATH
    )