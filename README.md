# AlphaChess 🤖♟️

AlphaChess is an AlphaZero-style Chess engine built with PyTorch, Monte Carlo Tree Search (MCTS), and Python. The project features a dual-headed policy and value neural network (AlphaChessNet), customized 19-channel chess board tensor representations, 4672-dimensional action-space mappings, multiple training pipelines (supervised, Stockfish distillation, and self-play reinforcement learning), and interactive frontends (both CLI and Pygame GUI).

---

## 🌟 Features

*   **Deep Residual Network Architecture:** A shared trunk ResNet with 10 residual blocks (configurable) powering two heads:
    *   **Policy Head:** Outputs move probabilities over $4672$ possible Queen-like, Knight, and pawn promotion moves.
    *   **Value Head:** Outputs a scalar evaluation $V(s) \in [-1, 1]$ representing the estimated outcome from the current player's perspective.
*   **Monte Carlo Tree Search (MCTS):** Implements tree expansion, PUCT selection, evaluation, and backpropagation for intelligent decision search.
*   **Three Robust Training Pipelines:**
    1.  **Supervised Training:** Train on large Lichess PGN game databases using standard and memory-safe streaming dataset classes.
    2.  **Stockfish Distillation:** Train the neural net to predict valuations and moves evaluated by the Stockfish chess engine.
    3.  **Self-Play Reinforcement Learning:** Train via reinforcement learning through self-play games guided by MCTS.
*   **Dual Interactive Interfaces:**
    *   **Pygame GUI:** An interactive chessboard UI with piece assets, move highlighting, and smart mouse inputs.
    *   **CLI Mode:** Play text-based chess directly inside the console using standard UCI notations.

---

## 📂 Project Architecture

```
AlphaChess/
│
├── core/                       # Core engine algorithms
│   ├── board_encoding.py       # Encodes chess.Board state into a 19-channel tensor
│   ├── move_encoding.py        # Maps chess.Move objects to/from 4672 policy indices
│   ├── model.py                # PyTorch neural network definition (AlphaChessNet)
│   ├── mcts.py                 # Monte Carlo Tree Search node & search logic
│   ├── self_play.py            # Generates training samples through self-play
│   ├── stockfish_trainer.py    # Teacher-student trainer wrapper using Stockfish
│   ├── trainer.py              # Supervised PGN learning engine
│   ├── config.py               # Hyperparameters and path configurations
│   └── utils.py                # Seeding, logging, and PyTorch helper functions
│
├── gui/                        # Pygame interface
│   ├── chess_gui.py            # Pygame chessboard and interactive loop
│   └── pieces/                 # Chess piece graphics (wp.png, bn.png, etc.)
│
├── scripts/                    # Run scripts
│   ├── play_game.py            # Play against the AI in terminal command line
│   ├── train_selfplay.py       # Script to launch MCTS self-play RL loop
│   ├── train_stockfish.py      # Script to launch Stockfish distillation training
│   └── train_supervised.py     # Script to train on historical PGN files
│
├── saved_models/               # Model checkpoints (.pth files)
├── data/                       # Directory for chess datasets (e.g. lichess_games.pgn)
├── logs/                       # Training logs
├── stockfish/                  # Directory for the Stockfish engine executable
└── requirements.txt            # Project dependencies
```

---

## 🛠️ Installation & Setup

### 1. Prerequisites
Ensure you have Python 3.8+ installed.

### 2. Clone and Setup Environment
Navigate to the project root and create a virtual environment:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Initialize Configurations
Initialize the directory structure and verify configs:
```bash
python core/config.py
```
This command automatically creates the `data/`, `saved_models/`, `logs/`, `stockfish/`, and `gui/pieces/` folders if they do not exist.

### 4. Setup Stockfish (Optional)
If you wish to run Stockfish distillation training:
1. Download Stockfish from [stockfishchess.org](https://stockfishchess.org/download/).
2. Place the executable inside the `stockfish/` directory.
3. Update `STOCKFISH_PATH` in `core/config.py` or `scripts/train_stockfish.py` to point to your specific executable.

---

## 🤖 Technical Deep Dive

### 1. Board Representation (19 Channels)
Every state is represented as a tensor of shape `(19, 8, 8)`:
*   **Channels 0-5:** Positions of active player's pieces (Pawns, Knights, Bishops, Rooks, Queens, Kings).
*   **Channels 6-11:** Positions of opponent player's pieces.
*   **Channel 12:** Side to move (all $1$s for White, all $0$s for Black).
*   **Channels 13-16:** Castling rights (White Kingside, White Queenside, Black Kingside, Black Queenside).
*   **Channel 17:** En passant square (if active, the single square is set to $1.0$).
*   **Channel 18:** Halfmove clock (scaled by $1/100$, capped at $1.0$).

### 2. Action Encoding (4672 Indices)
The policy head outputs a probability distribution over $4672$ actions:
$$\text{Policy Size} = 64 \times 73 = 4672$$
*   **64 Squares:** The source square `from_square` of the move.
*   **73 Move Planes:**
    *   **0-55:** Queen-like moves (8 directions $\times$ max distance of 7 squares).
    *   **56-63:** Knight moves (8 possible L-shape offsets).
    *   **64-72:** Pawn promotions (3 underpromotions to Knight, Bishop, Rook $\times$ 3 direction files: left-diagonal, straight, right-diagonal) + Queen promotion (which is represented as a Queen-like move).

### 3. MCTS & PUCT Selection Formula
During MCTS search, the selection phase uses the **PUCT (Predictor Upper Confidence bounds applied to Trees)** score to select actions:
$$\text{PUCT}(s, a) = Q(s, a) + U(s, a)$$
$$U(s, a) = C_{\text{puct}} \cdot P(s, a) \cdot \frac{\sqrt{\sum_b N(s, b) + 1}}{1 + N(s, a)}$$
*   $Q(s, a) = -V(s')$: Action value (estimated from the perspective of the active player).
*   $P(s, a)$: Prior probability of move $a$ given by the policy network.
*   $N(s, a)$: Visit count of the action.
*   $C_{\text{puct}}$: Constant controlling exploration vs. exploitation (defaults to $1.5$).

---

## 🚀 Running AlphaChess

### 🕹️ Interactive Play

#### Play in Pygame GUI
Play against the Chess AI using a graphical board.
```bash
python gui/chess_gui.py
```
*Note: Make sure your piece assets are downloaded and placed in `gui/pieces/` as `wp.png`, `bp.png`, etc. (e.g. standard Lichess/Chess.com style png files).*

#### Play in CLI Terminal
Play a text-based game against the engine in your console:
```bash
python scripts/play_game.py
```
Enter your moves in standard UCI coordinates (e.g., `e2e4`, `g1f3`).

---

### 🏋️ Training Pipelines

#### Pipeline A: Supervised Learning from PGNs
Trains AlphaChessNet on real human games.
1. Download a `.pgn` database (for example, Lichess database).
2. Place it in `data/lichess_games.pgn`.
3. Run the supervised script:
   ```bash
   python scripts/train_supervised.py
   ```
*   **Streaming Mode:** Set `STREAMING_DATASET = True` in configurations to lazily index game offsets in files. This enables training on massive PGN databases without filling up the system memory.

#### Pipeline B: Stockfish Distillation
Distill knowledge from Stockfish to bootstrap the model weights:
```bash
python scripts/train_stockfish.py
```
This generates random legal game board positions and trains the neural network to approximate Stockfish’s evaluations (Value Head) and selected best moves (Policy Head).

#### Pipeline C: Self-Play Reinforcement Learning
Train AlphaChess in an AlphaZero style:
```bash
python scripts/train_selfplay.py
```
The neural network generates games by playing against itself using MCTS, updates its weights based on the outcomes, and repeats the loop.

---

## 📊 Hyperparameters
Important training configurations are maintained in [core/config.py](file:///c:/Users/sesha/OneDrive/Desktop/AlphaChess/core/config.py):
*   `CHANNELS = 256` (Network Width)
*   `RESIDUAL_BLOCKS = 10` (Network Depth)
*   `MCTS_SIMULATIONS = 100` (Search Depth per turn)
*   `C_PUCT = 1.5` (MCTS exploration multiplier)
#   A l p h a C h e s s  
 #   A l p h a C h e s s  
 #   A l p h a C h e s s  
 