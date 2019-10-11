import torch
import torch.optim as optim
import torch.nn.functional as F
import os

from dataloader import Classifier
from model import Model
from utils import save_checkpoint

from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

writer = SummaryWriter()


# def test(data_loader, model, test_set):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     for data, target in data_loader:
#         data, target = data.to(device), target.to(device)
#         output = model(data)
#
#         # sum up batch loss
#         test_loss += F.nll_loss(output, target, size_average=False).item()
#         # get the index of the max log-probability
#         pred = output.max(1, keepdim=True)[1]
#         correct += pred.eq(target.view_as(pred)).sum().item()
#
#     test_loss /= len(data_loader.dataset)
#     accuracy = 100. * correct / len(data_loader.dataset)
#     print('\nTest set {} : Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
#           .format(test_set, test_loss, correct, len(data_loader.dataset),
#                   accuracy))
#     return accuracy


# def train(data_loader, model, epoch):
#     model.train()
#     batch_idx = 1
#     total_loss = 0
#     for images, labels in data_loader:
#         images = images.to(device)
#         labels = labels.to(device)
#         optimizer.zero_grad()
#         output = model(images)
#         loss = F.nll_loss(output, labels)
#         loss.backward()
#         optimizer.step()
#         if batch_idx % 10 == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(images), len(data_loader.dataset),
#                        100. * batch_idx / len(data_loader), loss.item()))
#         batch_idx = batch_idx + 1
#         total_loss = total_loss + loss.item()
#         writer.add_scalar('Loss/train', loss, epoch * len(data_loader) + batch_idx)
#     return total_loss


class DigitClassifier():
    def __init__(self, width=28, height=28, batch_size=64, learning_rate=0.0001, grayscale=True, shuffle=True):
        self.width = width
        self.height = height
        self.grayscale = grayscale
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.learning_rate = learning_rate
        self.model = Model().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def make_dataloader(self, data_dir):
        dataset = Classifier(root_dir=data_dir, width=self.width, height=self.height, grayscale=self.grayscale)
        dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                                 batch_size=self.batch_size,
                                                 shuffle=self.shuffle)
        return dataloader

    def train_epoch(self, data_loader, epoch):
        self.model.train()
        batch_idx = 1
        total_loss = 0
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            self.optimizer.zero_grad()
            output = self.model(images)
            loss = F.nll_loss(output, labels)
            loss.backward()
            self.optimizer.step()
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(images), len(data_loader.dataset),
                           100. * batch_idx / len(data_loader), loss.item()))
            batch_idx = batch_idx + 1
            total_loss = total_loss + loss.item()
            writer.add_scalar('Loss/train', loss, epoch * len(data_loader) + batch_idx)
        return total_loss

    def test(self,data_loader):
        self.model.eval()
        test_loss = 0
        correct = 0
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = self.model(data)

            # sum up batch loss
            test_loss += F.nll_loss(output, target, size_average=False).item()
            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(data_loader.dataset)
        accuracy = 100. * correct / len(data_loader.dataset)
        # print('\nTest set {} : Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
        #       .format(test_set, test_loss, correct, len(data_loader.dataset),
        #               accuracy))
        return accuracy,test_loss


if __name__ == "__main__":

# training_data = './data/'
# test1 = './test1/'
# test2 = './test2/'
# width = 28
# height = 28
# grayscale = True
# train_digit_dataset = Classifier(root_dir=training_data, width=width, height=height, grayscale=grayscale)
# train_data_loader = torch.utils.data.DataLoader(dataset=train_digit_dataset,
#                                                 batch_size=64,
#                                                 shuffle=True)
# test_dataset_1 = Classifier(root_dir=test1, width=width, height=height, grayscale=grayscale)
# test_loader_1 = torch.utils.data.DataLoader(dataset=test_dataset_1,
#                                             batch_size=256,
#                                             shuffle=True)
# test_dataset_2 = Classifier(root_dir=test2, width=width, height=height, grayscale=grayscale)
# test_loader_2 = torch.utils.data.DataLoader(dataset=test_dataset_2,
#                                             batch_size=256,
#                                             shuffle=True)
# model = Model().to(device)
# optimizer = optim.Adam(model.parameters(), lr=0.0001)
#
# best_avg_accuracy = 0.0
# best_training_loss = 0.0
# for epoch in range(1, 100):
#     training_loss = train(train_data_loader, model, epoch)
#     accuracy1 = test(test_loader_1, model, test1)
#     accuracy2 = test(test_loader_2, model, test2)
#     if (accuracy1 + accuracy2) / 2 > best_avg_accuracy:
#         best_avg_accuracy = (accuracy1 + accuracy2) / 2
