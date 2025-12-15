from datetime import datetime

#<------------------------------------------------------------------------------------------------>
train_config = dict(
    debugging = False,
    checkpoint_path = 
    'tb_logs/bbdm_adspainn/final/checkpoints/scorenet-epoch=3999-avg_val_loss=0.047.ckpt',
    flow = 'bbdm', # [bbdm], rf
    coord = 'cartesian', # [cartesian], fractional
    epoch = 2000, # [2000]
    model_name = 'adspainn', # [painn], egnn, adspainn, equiformerv2
    batch_size = 128, # [128] * 4
    lr = 1e-4, # [1e-4]
    schedule_gamma = 0.998,
    num_workers = 0,
    matmul_precision = 'highest',
    clip_grad = True,
    loss_type = 'l1', # l2 or [l1] (l1 for bbdm, l2 for rf)
    fixed = True, # whether to fix the atoms below
    train_objective = 'grad', # grad, noise, ysubx (only for bbdm)
    frac_noise = True, # auto False for 'cartesian' coord mode
    write_outputs = False,
    ema = False,
    ema_decay = 0.99,
)

#<------------------------------------------------------------------------------------------------>
data_config = dict(
    path = 'data/oc20/clean/metal/', # path to the data
)

#<------------------------------------------------------------------------------------------------>
utils_config = dict(
    timepoint = datetime.now().strftime("%m-%d-%H:%M:%S"),
    prefix = 'ft_all_', # default is empty
    sample_per_epoch = 5, 
    num_timesteps = 1000, 
    sample_mt_mode = 'cosine', # linear, sin, cosine
    max_var = 0.05, # 0.001 ~ 0.05
    sample_steps = 250, 
    skip_sample = True, # True, False
    sample_mode = 'linear', # linear, cosine, all
    eta = 0.,
)
