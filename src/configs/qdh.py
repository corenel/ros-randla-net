# -*- coding: utf-8 -*-


class ConfigQDH:
    root = '/datasets/data/ROI_scan'  # train data
    test_root = '/datasets/data/raw_scan'  # test_data
    k_n = 16  # KNN
    num_layers = 4  # Number of layers
    num_points = 4096  # Number of input points
    num_classes = 2  # Number of valid classes
    sub_grid_size = 0.06  # preprocess_parameter

    batch_size = 8  # batch_size during training
    num_workers = 16
    # val_batch_size = 20  # batch_size during validation and test
    # train_steps = 500  # Number of steps per epochs
    # val_steps = 100  # Number of validation steps per epoch

    sub_sampling_ratio = [4, 4, 4, 4]
    d_out = [16, 64, 128, 256]  # feature dimension
    num_sub_points = [
        num_points // 4, num_points // 16, num_points // 64, num_points // 256
    ]

    noise_init = 3.5  # noise initial parameter
    max_epoch = 100  # maximum epoch during training
    learning_rate = 1e-2  # initial learning rate
    # lr_decays = {i: 0.95 for i in range(0, 500)}  # decay rate of learning rate
    step_size = 1
    decay_rate = 0.95
    delta_d = 1.5
    delta_v = 0.5
    bandwidth = 1.5
    initial_temperature = 1.0
    weight_decay = 1e-6

    saving = True
    saving_path = None
    train = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10]
    valid = [8]

    def __init__(self):
        pass
