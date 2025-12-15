from datetime import datetime
from model.net import PaiNN, EGNN, AdsPaiNN, FTPaiNN

#<------------------------------------------------------------------------------------------------>
train_config = dict(
    debugging = False,
    flow = 'bbdm', # [bbdm], rf
    coord = 'cartesian', # [cartesian], fractional
    epoch = 2000, # [4000]
    model_name = 'adspainn', # [painn], egnn, adspainn, equiformerv2
    batch_size = 128, # [128] * 4
    lr = 1e-4, # [1e-3]
    schedule_gamma = 0.999,
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
if train_config['model_name'] == 'painn':
    # painn
    model_config = dict(
        model = PaiNN if train_config['coord'] == 'cartesian' else FTPaiNN,
        cutoff = 4.0,
        hidden_channels = 256,
        out_channels = 3,
        num_rbf = 256,
        rbf = {'name': 'gaussian'},
        envelope = {'name': 'polynomial', 'exponent': 5},
        num_layers = 4,
        n_frequencies = 40, # for fourier features
        scalar = False, # True or [False], using scalar xh_out or not
        ftbasis = False, # [True] or False, using fourier basis or not
    )
elif train_config['model_name'] == 'egnn':
    # egnn
    model_config = dict(
        model = EGNN,
        hidden_dim = 256,
        latent_dim = 256,
        num_layers = 4,
        max_atoms = 100,
        act_fn = 'silu',
        dis_emb = 'sin',
        num_freqs = 40,
        ln = False,
        pred_scalar = False, # True(invariant), False(equivariant)
    ) 
elif train_config['model_name'] == 'adspainn':
    # adspainn
    model_config = dict(
        model = AdsPaiNN if train_config['coord'] == 'cartesian' else FTPaiNN,
        cutoff = 4.0,
        hidden_channels = 256,
        out_channels = 3,
        num_rbf = 256,
        rbf = {'name': 'gaussian'},
        envelope = {'name': 'polynomial', 'exponent': 5},
        num_layers = 4,
        n_frequencies = 40, # for fourier features
        scalar = False, # True, False, using scalar xh_out or not
        ftbasis = False, # [True] or False, using fourier basis or not
    )
elif train_config['model_name'] == 'equiformerv2':
    # equiformerv2
    raise NotImplementedError(f"model_name {train_config['model_name']} not supported")
else:
    raise ValueError(f"model_name {train_config['model_name']} not supported")

#<------------------------------------------------------------------------------------------------>
data_config = dict(
    path = 'data/cathub/', # path to the data
)

#<------------------------------------------------------------------------------------------------>
utils_config = dict(
    timepoint = datetime.now().strftime("%m-%d-%H:%M:%S"),
    prefix = '', # default is empty
    sample_per_epoch = 5,
    num_timesteps = 100,
    sample_mt_mode = 'cosine', # linear, sin, cosine
    max_var = 0.05, # 0.001 ~ 0.05
    sample_steps = 20, 
    skip_sample = True, # True, False
    sample_mode = 'linear', # linear, cosine, all
    eta = 0.,
)
