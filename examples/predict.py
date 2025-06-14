import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from bimigan.torch.funcs import get_torch_device_smart, seed_everything
from bimigan.config import Args
from bimigan.cgan import construct_model
from bimigan.data.datasets.cyclegan import CycleGANDataset

def load_gen_ab(model_path, checkpoint_epoch, device):
    args   = Args.load(model_path)
    model = construct_model(
    args.savedir, args.config, is_train = False, device = device
    )
    if checkpoint_epoch == -1:
        checkpoint_epoch = max(model.find_last_checkpoint_epoch(), 0)

    print("Load checkpoint at epoch %s" % checkpoint_epoch)

    seed_everything(args.config.seed)
    model.load(checkpoint_epoch)
    gen_ab = model.models.gen_ab
    ##gen_ab.to('cpu')
    return gen_ab

def making_predictions(dataloader,gen_ab):
    gen_ab.eval()
    for (inputA, _) in dataloader:
        with torch.no_grad():
            features = gen_ab(inputA)
        file = features.detach().cpu().numpy()
        print(file.ndim)
        file_save = file   ##file.squeeze()
        print(file_save.ndim)
        plt.imsave(
            "/workspace/data/result/",
            file_save)  ##,cmap='gray')


def create_data_loader(data_path, batch_size):
    # Change or add transformations as per your needs
    transformations = [
        # transforms.CenterCrop((224, 224)),
        #transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ]
    ds = CycleGANDataset(
        data_path,
        is_train=False,
        transform = transforms.Compose(transformations))
    dl = DataLoader(ds, batch_size=batch_size,shuffle=False)
    return dl

if __name__ == '__main__':
    BATCH_SIZE = 1
    EPOCH      = -1
    DEVICE     = get_torch_device_smart()
    DATA_PATH  =  "/workspace/data/selfie2anime/"# "<path_to_your_data>"
    # Specify the path of your model trained through BiMIGAN.
    MODEL_PATH = "/workspace/bimigan/bimigan/outdir/selfie2anime/model_d(cyclegan)_m(cyclegan)_d(basic)_g(vit-unet)_cyclegan_vit-unet-12-self-lsgan-paper-cycle_high-256" ##"<path_to_saved_model>"
    trained_model = load_gen_ab(MODEL_PATH, EPOCH, DEVICE)
    DL = create_data_loader(DATA_PATH, BATCH_SIZE)
    making_predictions(DL,trained_model)
    print("Finish")
