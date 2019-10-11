import torch
import shutil
import os


def save_model(model, optimizer, epoch, training_loss, save_path='./saved_models'):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    filename = os.path.join(save_path, str('ep' + str(epoch) + '.pth.tar'))
    torch.save({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'training_loss': training_loss
    }, filename)
