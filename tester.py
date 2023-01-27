import torchvision
from model.model import LeNet
from config import cifar10_config, sr_config
from utils import imshow, psnr
import torch
from model.zssr import ZSSR
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def test_one_batch(parameters):
    net = LeNet()
    net.load_state_dict(torch.load(cifar10_config.SAVEPATH))
    classes = parameters['class_label']
    net.eval()

    dataiter = iter(parameters['test_loader'])
    images, labels = next(dataiter)

    # print images
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

def test(parameters):
    net = LeNet()
    net.load_state_dict(torch.load(cifar10_config.SAVEPATH))
    classes = parameters['class_label']
    net.eval()

    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for data in parameters['test_loader']:
            images, labels = data
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

def test_zssr(device):
    net = ZSSR()
    net.load_state_dict(torch.load(sr_config.SAVE_PATH))
    net.eval()

    testimg = Image.open(sr_config.TEST_IMAGE_PATH).convert('RGB')
    testimg_in = np.array(testimg).astype(np.float32)/255.
    testimg_in = np.expand_dims(testimg_in,0)
    testimg_in = torch.tensor(np.transpose(testimg_in, (0,3,1,2)))
    testimg_in.to(device)

    sr_img = net(testimg_in)
    sr_img = sr_img.cpu().detach().numpy()
    sr_img = np.transpose(sr_img, (0,2,3,1))
    sr_img = sr_img[0,:,:,:]
    sr_img = np.uint8(np.clip(np.round(sr_img*255.),0,255))

    score = psnr(sr_img, testimg)
    print(score)

    fig = plt.figure(figsize=(10,10))

    plt.subplot(1,2,1)
    plt.imshow(testimg)
    plt.title('GT image')

    plt.subplot(1,2,2)
    plt.imshow(sr_img)
    plt.title('SR image')

    plt.show()

