# Dataset Preparation

Download the datasets from [link]()

## Dataset Structure
```bash
data/
├── bifurcating_hill/
│   ├── all/
│   ├── test/
│   ├── train/
│   ├── transforms_train.json
│   └── transforms_test.json
├── oscillating_ball/
│   ├── all/
│   ├── test/
│   ├── train/
│   ├── transforms_train.json
│   └── transforms_test.json
```

# Training

<details>
<summary><strong>Bifurcating Hill</strong></summary>

```bash
python run_lips.py --config configs/config_bifurcating_hill.txt
```
</details>

<details>
<summary><strong>Oscillating Ball</strong></summary>

```bash
python run_lips.py --config configs/config_oscillating_ball.txt
```
</details>

# Testing

<details>
<summary><strong>Bifurcating Hill</strong></summary>

```bash
python render.py --config configs/config_bifurcating_hill.txt --ckpt_path [CHECKPOINT PATH]
```
</details>

<details>
<summary><strong>Oscillating Ball</strong></summary>

```bash
python render.py --config configs/config_oscillating_ball.txt --ckpt_path [CHECKPOINT PATH]
```
</details>
