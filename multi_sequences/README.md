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

```bash
python run_lips.py --config configs/config.txt
```
# Testing
