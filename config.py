class cifar10_config:
    #  parameter
    EPOCH = 10
    SAVEPATH = './checkpoint/cifar_net.pth'
    BATCH_SIZE = 4

class sr_config:
    IMAGE_PATH = './data/Set5'
    OUTPUT_PATH = './data/dataset.h5'
    SAVE_PATH = 'checkpoint/zssr_bak.pth'
    SCALE = 2
    PATCH_SIZE = 64
    STRIDE = 20
    BATCH_SIZE = 64
    EPOCH = 20
    TEST_IMAGE_PATH = './data/Set5/butterfly.png'
