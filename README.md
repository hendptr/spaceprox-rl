# SpaceProx Black-Box RL

SpaceProx is a small Direct3D 11 arena game used as a black-box reinforcement
learning testbed. The game and RL controller are separate Windows processes.

The RL code does not read process memory, inject a DLL, or call an internal game API.
It captures the rendered game window, extracts visual features, and sends ordinary
keyboard and mouse input.

## Project layout

```text
SpaceProxPackage/
|-- SpaceProx/
|   |-- SpaceProx.exe
|   |-- SpaceProx.cpp
|   |-- Player.h
|   |-- PlayerFactory.cpp
|   `-- SpaceProx.vcxproj
|-- RL/
|   |-- game_interface.py
|   |-- final_env.py
|   |-- final_model.py
|   |-- instrumented_ppo.py
|   |-- train_final.py
|   |-- play_final.py
|   |-- probe_final.py
|   |-- evaluate_final.py
|   `-- requirements.txt
|-- models/
|   `-- spaceprox_best.zip
|-- install.bat
|-- run_best.bat
|-- train_5min.bat
`-- train_30min.bat
```

## Requirements

- Windows 10 or Windows 11
- Python 3.12 recommended
- NVIDIA GPU with a CUDA-capable PyTorch installation for the supplied scripts
- Visual Studio 2022 only if you want to rebuild the game

## Quick start

### Install Python dependencies

Double-click `install.bat`, or run:

```bat
cd RL
python -m pip install -r requirements.txt
```

### Play the supplied model

Double-click:

```text
run_best.bat
```

The Python environment launches `SpaceProx\SpaceProx.exe` automatically when the
game is not already open.

The PPO policy learns movement only. A deterministic visual controller handles enemy
detection, target tracking, mouse aim, and firing.

Only run one controller/game pair at a time because keyboard state and cursor position
are global Windows input. Press `Ctrl+C` in the Python terminal to stop playback.

### Train a new model

For a five-minute experiment:

```text
train_5min.bat
```

For a thirty-minute experiment:

```text
train_30min.bat
```

Each training run writes to its own checkpoint and log directories under `RL`.

## Evaluate a checkpoint

From the `RL` directory:

```bat
python probe_final.py --model ..\models\spaceprox_best.zip --steps 1000
```

For multi-episode evaluation:

```bat
python evaluate_final.py --model ..\models\spaceprox_best.zip --episodes 10
```

The behavior probe reports movement diversity, policy entropy, arena coverage,
blocked movement, stagnation, corner occupancy, score, and HP.

## Build SpaceProx from source

The repository already includes `SpaceProx\SpaceProx.exe`, so rebuilding is optional.

Example with Visual Studio 2022 Professional:

```powershell
& "C:\Program Files\Microsoft Visual Studio\2022\Professional\MSBuild\Current\Bin\amd64\MSBuild.exe" `
  .\SpaceProx\SpaceProx.vcxproj /t:Rebuild /p:Configuration=Release /p:Platform=x64
```

The rebuilt executable is written to `SpaceProx\bin\x64\Release\SpaceProx.exe`.
Copy it to `SpaceProx\SpaceProx.exe` if you want the Python launcher to use it.

## Game controls

| Input | Action |
|---|---|
| `WASD` / arrow keys | Move |
| Mouse | Aim |
| `Space` | Fire |
| `P` | Pause |
| `R` / `Enter` | Retry after game over |
| `Esc` | Quit |

## What the model learns

The PPO network receives a 76-dimensional observation built from rendered-image
measurements, short movement history, and the previous movement action. It selects one
of nine movement commands:

```text
idle, W, S, A, D, W+A, W+D, S+A, S+D
```

The game remains a separate Windows process throughout training and playback.
