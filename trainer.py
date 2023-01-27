import torch
import os
from tqdm import tqdm

def train(dataloader, model, epochs, savepath):

    model['net'].train()
    print('Start Training...')

    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        min_loss = 100
        pbar = tqdm(enumerate(dataloader), total=len(dataloader))

        for i, data in pbar:
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(model['device'])
            labels = labels.to(model['device'])

            # zero the parameter gradients
            model['optimizer'].zero_grad()

            # forward + backward + optimize
            outputs = model['net'](inputs)
            loss = model['criterion'](outputs, labels)
            loss.backward()
            model['optimizer'].step()

            # print statistics
            running_loss += loss.item()
            pbar.set_postfix({'epoch':epoch+1, 'loss': running_loss / (i+1)})
            # if i % 2000 == 1999:    # print every 2000 mini-batches
            #     print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            #     running_loss = 0.0

            # save
            if loss.item() <= min_loss:
                os.makedirs('./checkpoint', exist_ok=True)
                torch.save(model['net'].state_dict(), savepath)
                min_loss = loss.item()

    print('Finished Training')


