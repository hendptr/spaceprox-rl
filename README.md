# SpaceProx Black-Box RL

This folder contains the standalone SpaceProx game and the final black-box PPO
controller used in the accompanying article.

The RL code does not read game memory, inject a DLL, or call an internal game API.
It captures the rendered client area, extracts visual features, and controls the game
with ordinary Windows keyboard and mouse input.

## Folder layout

```text
SpaceProxPackage/
├── Game/
│   ├── SpaceProx.exe
│   ├── SimpleGame7.cpp
│   ├── Player.h
│   ├── PlayerFactory.cpp
│   └── SpaceProx.vcxproj
├── RL/
│   ├── game_interface.py
│   ├── final_env.py
│   ├── final_model.py
│   ├── instrumented_ppo.py
│   ├── train_final.py
│   ├── play_final.py
│   ├── probe_final.py
│   ├── evaluate_final.py
│   └── requirements.txt
├── models/
│   └── spaceprox_best.zip
├── install.bat
├── run_best.bat
├── train_5min.bat
└── train_30min.bat
```

## Requirements

- Windows 10 or Windows 11
- Python 3.12 recommended
- NVIDIA GPU with a CUDA-capable PyTorch installation for the supplied scripts
- Visual Studio 2022 only if you want to rebuild the game

## Quick start

### 1. Install Python dependencies

Double-click:

```text
install.bat
```

or run:

```bat
cd RL
python -m pip install -r requirements.txt
```

### 2. Play with the supplied trained model

Double-click:

```text
run_best.bat
```

You do not need to start the game manually. The Python environment launches
`Game\SpaceProx.exe` automatically if it is not already open.

The learned PPO policy chooses movement only. A deterministic visual controller
handles enemy detection, target tracking, mouse aim, and firing.

### 3. Train a new model

For a quick five-minute experiment:

```text
train_5min.bat
```

For a longer thirty-minute experiment:

```text
train_30min.bat
```

Each training run uses a unique output directory under `RL\checkpoints\final` and
`RL\logs\final`.

## Evaluate a checkpoint

From the `RL` directory:

```bat
python probe_final.py --model ..\models\spaceprox_best.zip --steps 1000
```

For full episode evaluation:

```bat
python evaluate_final.py --model ..\models\spaceprox_best.zip --episodes 10
```

The behavior probe reports movement diversity, entropy, arena coverage, blocked
movement, stagnation, corner occupancy, score, and HP.

## Build SpaceProx from source

The package already includes a compiled `Game\SpaceProx.exe`, so rebuilding is
optional.

Example with Visual Studio 2022 Professional:

```powershell
& "C:\Program Files\Microsoft Visual Studio\2022\Professional\MSBuild\Current\Bin\amd64\MSBuild.exe" `
  .\Game\SpaceProx.vcxproj /t:Rebuild /p:Configuration=Release /p:Platform=x64
```

If your Visual Studio edition is Community or Enterprise, adjust the MSBuild path.

## Controls

| Input | Action |
|---|---|
| `WASD` / arrow keys | Move |
| Mouse | Aim |
| `Space` | Fire |
| `P` | Pause |
| `R` / `Enter` | Retry after game over |
| `Esc` | Quit |

## What the model learns

The PPO network receives a 76-dimensional observation made from rendered-image
measurements, short movement history, and the previous movement action. It outputs
one of nine movement commands:

```text
idle, W, S, A, D, W+A, W+D, S+A, S+D
```

The game remains a separate Windows process throughout training and playback.

