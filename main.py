from data.dataloader import load_cifar10_dataset, TrainDataset
from trainer import train
from tester import test_one_batch, test, test_zssr
import torch.optim as optim
from model.model import LeNet
import torch.nn as nn
import torch
from config import cifar10_config, sr_config
from data.gen_dataset import prepare, showimage
import utils
import glob

from model.zssr import ZSSR

def cifar10():
    net = LeNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    net.to(device)
    model = {
        'net': net,
        'optimizer': optimizer,
        'criterion': criterion,
        'device': device
    }

    dataset = load_cifar10_dataset()
    train(dataset, model, cifar10_config.EPOCH, cifar10_config.SAVEPATH)
    test_one_batch(dataset, model)
    test(dataset, model)

def train_zssr():
    net = ZSSR()
    net.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    model = {
        'net': net,
        'optimizer': optimizer,
        'criterion': criterion,
        'device': device
    }
    dataset = TrainDataset(sr_config.OUTPUT_PATH)
    dataloader = torch.utils.data.DataLoader(dataset,sr_config.BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)
    # data = {'train_loader':dataloader}

    train(dataloader, model, sr_config.EPOCH, sr_config.SAVE_PATH)


def basic_gen_dataset():
    # super resolution process
    prepare(sr_config.IMAGE_PATH, sr_config.OUTPUT_PATH, sr_config.SCALE, sr_config.PATCH_SIZE, sr_config.STRIDE)
    showimage(sr_config.OUTPUT_PATH)

def random_argument_test():
    prepare(sr_config.IMAGE_PATH, sr_config.OUTPUT_PATH, sr_config.SCALE, sr_config.PATCH_SIZE, sr_config.STRIDE, True)
    showimage(sr_config.OUTPUT_PATH)



if __name__ == '__main__':

    device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')
    torch.manual_seed(777)

    # cifar10()
    # basic_gen_dataset()
    # random_argument_test()
    # train_zssr()
    test_zssr(device)








