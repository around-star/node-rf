
# Training

Bouncing Balls Scene
```bash
python train_warmup.py --config configs/bouncingballs.txt
```

DyNeRF Dataset
```bash
python train_warmup.py --config configs/configs_dynerf.txt --num_cams 18
```


# Testing

Bouncing Balls Scene
```bash
python render.py --config configs/bouncingballs.txt --ckpt_path [CHECKPOINT PATH]
```

DyNeRF Dataset
```bash
python render.py --config configs/configs_dynerf.txt --num_cams 18 --ckpt_path [CHECKPOINT PATH]
```

### Important args to consider
--num_obs : The initial number of observations to learn as latents

--render_num : The number of rendering time steps

--latent_dim : The dimension of each latent

--ode_type : To choose the ode solver

--sample_size : The length of the training sequence
