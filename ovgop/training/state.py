"""Restore training settings that change during the epoch schedule."""


def restore_training_loss_factors(args, checkpoint):
    saved_args = checkpoint.get("args")
    if saved_args is None:
        saved_args = {}
    elif not isinstance(saved_args, dict):
        saved_args = vars(saved_args)

    transition_epoch = saved_args.get("ovd_frozen_epoch", args.ovd_frozen_epoch)
    after_transition = checkpoint["epoch"] >= transition_epoch
    defaults = {
        "loss_amp_factor": 6000 if after_transition else args.loss_amp_factor,
        "loss_energy_factor": 60 if after_transition else args.loss_energy_factor,
    }
    for name, default in defaults.items():
        setattr(args, name, saved_args.get(name, default))
