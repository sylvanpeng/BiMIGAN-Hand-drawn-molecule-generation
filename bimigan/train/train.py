from itertools import islice
import tqdm
import torch
from bimigan.config import Args
from bimigan.data import get_data
from bimigan.torch.funcs import get_torch_device_smart, seed_everything
from bimigan.cgan import construct_model
from bimigan.utils.log import setup_logging

from bimigan.train.metrics import LossMetrics
from bimigan.train.callbacks import TrainingHistory
from .transfer  import transfer



def training_epoch(it_train, model, title, steps_per_epoch):
    steps = len(it_train)
    if steps_per_epoch is not None:
        steps = min(steps, steps_per_epoch)

    progbar = tqdm.tqdm(desc=title, total=steps, dynamic_ncols=True)
    metrics = LossMetrics()

    for batch in islice(it_train, steps):
        model.set_input(batch)
        model.optimization_step()

        metrics.update(model.get_current_losses())

        progbar.set_postfix(metrics.values, refresh=False)
        progbar.update()

    progbar.close()
    return metrics


def try_continue_training(args, model):
    history = TrainingHistory(args.savedir)

    start_epoch = model.find_last_checkpoint_epoch()
    model.load(start_epoch)

    if start_epoch > 0:
        history.load()

    start_epoch = max(start_epoch, 0)

    return (start_epoch, history)


def train(args_dict):
    args = Args.from_args_dict(**args_dict)

    setup_logging(args.log_level)
    seed_everything(args.config.seed)
    device = get_torch_device_smart()
    it_train, _ = get_data(
        args.config.data, args.config.batch_size, args.workers
    )

    print("Starting training...")
    print(args.config.to_json(indent=4))

    model = construct_model(
        args.savedir, args.config, is_train=True, device=device
    )
    start_epoch, history = try_continue_training(args, model)

    if (start_epoch == 0) and (args.transfer is not None):
        transfer(model, args.transfer)

    for epoch in range(start_epoch + 1, args.epochs + 1):
        title = 'Epoch %d / %d' % (epoch, args.epochs)
        metrics = training_epoch(
            it_train, model, title, args.config.steps_per_epoch
        )

        history.end_epoch(epoch, metrics)
        model.end_epoch(epoch)

        if epoch % args.checkpoint == 0:
            model.save(epoch)

    model.save(epoch=None)
