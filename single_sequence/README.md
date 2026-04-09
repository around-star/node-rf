
# Dataset Preparation

<details>
<summary><strong>Bouncing Balls Scene</strong></summary>
  
Follow the original setup used in [D-NeRF](https://github.com/albertpumarola/D-NeRF).

</details>

<details>
<summary><strong>DyNeRF Dataset</strong></summary>

Download a scene from [DyNeRF](https://github.com/facebookresearch/Neural_3D_Video/releases/tag/v1.0).

Extract the frames from the videos and store all the frames ```[DATASET PATH]/images```, naming the frames 0.png, 1.png, ...., 300.png, 301.png, ....

</details>


## Dataset Structure

```bash
data/
├── bouncingballs/
│   ├── test/
│   ├── train/
│   ├── val/
│   ├── transforms_train.json
│   ├── transforms_test.json
│   └── transforms_val.json
├── sear_steak/
│   ├── images/
│   ├── poses_bounds.npy
│   ├── cam_00.mp4
│   ├── cam_01.mp4
│   └── ...
```

# Training

<details>
<summary><strong>Bouncing Balls Scene</strong></summary>

```bash
python train_warmup.py --config configs/bouncingballs.txt
```
</details>

<details>
<summary><strong>DyNeRF Dataset</strong></summary>
  
```bash
python train_warmup.py --config configs/config_dynerf.txt --num_cams 18 --datadir [DATASET PATH]
```
</details>


# Testing

<details>
<summary><strong>Bouncing Balls Scene</strong></summary>
  
```bash
python render.py --config configs/bouncingballs.txt --ckpt_path [CHECKPOINT PATH]
```
</details>

<details>
<summary><strong>DyNeRF Dataset</strong></summary>

```bash
python render.py --config configs/configs_dynerf.txt --num_cams 18 --datadir [DATASET PATH] --ckpt_path [CHECKPOINT PATH]
```
</details>

## Important args to consider

--num_obs : The initial number of observations to learn as latents

--render_num : The number of rendering time steps

--latent_dim : The dimension of each latent

--ode_type : To choose the ode solver

--sample_size : The length of the training sequence
