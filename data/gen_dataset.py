import glob
import PIL.Image as pil_image
import h5py
import numpy as np
import utils

def prepare(image_path, output_path, scale, patch_size, stride, rand_aug=False):
    hr_patches = []
    lr_patches = []

    for file in glob.iglob(image_path + '/*.png', recursive=True):
        hr = pil_image.open(file).convert('RGB')

        if rand_aug is True:
            hr = utils.random_agument(hr)
            hr = np.uint8(np.clip(np.round(hr*255.),0,255))
            hr = pil_image.fromarray(hr)
        # mode crop for fitting scale
        hr_width = (hr.width // scale) * scale
        hr_height = (hr.height // scale) * scale
        hr = hr.crop((0,0,hr_width,hr_height)) # left, upper, right, lower

        # image scale down for lr with BICUBIC interporlation
        lr = hr.resize((hr.width // scale, hr.height // scale), resample=pil_image.BICUBIC)
        # scale restore for ilr
        ilr = lr.resize((lr.width * scale, lr.height * scale), resample=pil_image.BICUBIC)
        # scaling
        hr = np.array(hr).astype(np.float32) / 255.
        ilr = np.array(ilr).astype(np.float32) / 255.
        # hr = np.array(hr)
        # ilr = np.array(ilr)

        # crop paired patch
        for i in range(0, ilr.shape[0]-patch_size+1, stride):
            for j in range(0, ilr.shape[1]-patch_size+1, stride):
                lr_patches.append(ilr[i:i+patch_size, j:j+patch_size, :])
                hr_patches.append(hr[i:i+patch_size, j:j+patch_size, :])

    # image patch -> array
    lr_patches = np.array(lr_patches)
    hr_patches = np.array(hr_patches)
    lr_patches = np.transpose(lr_patches, (0, 3, 1, 2))
    hr_patches = np.transpose(hr_patches, (0, 3, 1, 2))
    print(f'hr patches shape : {hr_patches.shape} lr patches shape : {lr_patches.shape}')

    # create h5 file
    h5_file = h5py.File(output_path, 'w')
    h5_file.create_dataset('lr', data=lr_patches)
    h5_file.create_dataset('hr', data=hr_patches)

    h5_file.close()

def showimage(data_path):
    h5_file = h5py.File(data_path, 'r')
    images = np.array(h5_file['hr']) * 255.
    images = images.astype(np.uint8)
    images = np.transpose(images, (0,2,3,1))
    print(f'show image shape : {images[0].shape}')
    img = pil_image.fromarray(images[0], 'RGB')
    pil_image._show(img)
