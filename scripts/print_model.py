#!/usr/bin/env python

import sys

from bimigan.config      import Args
from bimigan.torch.funcs import get_torch_device_smart
from bimigan.cgan        import construct_model

def main(path):
    args   = Args.load(path)
    device = get_torch_device_smart()
    model  = construct_model(
        args.savedir, args.config, is_train = True, device = device
    )

    epoch = max(model.find_last_checkpoint_epoch(), 0)
    print("Load checkpoint at epoch %s" % epoch)

    model.load(epoch)

    print(model.pprint(verbose = True))

if __name__ == '__main__':
    main(sys.argv[1])

