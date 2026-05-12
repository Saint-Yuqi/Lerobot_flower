"""Flower-env-native FlowerVLA stack.

Runs entirely in the `flower` conda env (Python 3.10, torch 2.2.2,
transformers 4.46). Independent of the lerobot package because lerobot 0.5.x
requires Python 3.12 and the SmolVLA bits pin transformers 5.x.

Modules:
    dataset.py     — SO-101 LeRobotDataset v3.0 reader (parquet + mp4)
    normalizer.py  — per-dim normalize/unnormalize for action + state
    policy.py      — FlowerVLAPolicy wrapping the vendored FlowerVLA model
    runner.py      — closed-loop action runner for real-robot inference
"""
