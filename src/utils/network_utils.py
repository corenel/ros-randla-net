import torch
import os
import torch.optim as optim

from models.RandLANet import Network


def load_network(config, device, resume_path=None):
    if resume_path is not None and os.path.exists(resume_path):
        model, optimizer, start_epoch = resume(config, resume_path)
    else:
        model = Network(config)
        model.to(device)
        start_epoch = 0
        parameters = model.parameters()
        optimizer = optim.Adam(parameters, lr=config.learning_rate)

    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                          config.step_size,
                                          gamma=config.decay_rate)

    return model, optimizer, start_epoch, scheduler


def resume(config, resume_pth):
    """ Loads model and optimizer state from a previous checkpoint. """
    print("=> loading checkpoint '{}'".format(resume_pth))
    checkpoint = torch.load(resume_pth)

    model = Network(config)
    model.load_state_dict(checkpoint['state_dict'], False)
    model.to('cuda')

    parameters = model.parameters()
    optimizer = optim.Adam(parameters, lr=config.learning_rate)
    if 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    for group in optimizer.param_groups:
        group['initial_lr'] = config.learning_rate

    start_epoch = checkpoint['epoch']

    return model, optimizer, start_epoch
