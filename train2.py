import torch
import torch.optim as optim
import torch.nn.functional as F
import os
from datetime import datetime

from dataloader import Classifier
from models.vgg16 import Model

# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)
# exit()
torch.manual_seed(42)
  

class DigitClassifier(object):
    def __init__(self, width=28, height=28, batch_size=32, learning_rate=0.0005, grayscale=True, shuffle=True):
        x=28
        y=28
        self.width = 30
        self.height = 40
        self.grayscale = grayscale
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.learning_rate = learning_rate
        self.model = Model().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.training_loss = SummaryWriter()
        self.validation_acc = SummaryWriter()

    def make_dataloader(self, data_dir):
        dataset = Classifier(root_dir=data_dir, height=self.height, width=self.width, grayscale=self.grayscale)
        dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                                 batch_size=self.batch_size,
                                                 shuffle=self.shuffle)
        return dataloader

    def save_model(self, epoch, training_loss, save_path):
        filename = os.path.join(save_path, str('ep' + str(epoch) + '.pth.tar'))
        torch.save({
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'training_loss': training_loss
        }, filename)
        print('Model saved successfullt at epoch : ', epoch)

    def train_epoch(self, data_loader, epoch):
        self.model.train()
        batch_idx = 1
        total_loss = 0
        for images, labels in data_loader:
            # print(images.size())
            images = images.to(device)
            labels = labels.to(device)
            self.optimizer.zero_grad()
            output = self.model(images)
            # break
            loss = F.cross_entropy(output, labels)
            loss.backward()
            self.optimizer.step()
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(images), len(data_loader.dataset),
                           100. * batch_idx / len(data_loader), loss.item()))
            batch_idx = batch_idx + 1
            total_loss = total_loss + loss.item()
            self.training_loss.add_scalar('Loss/train', loss, epoch * len(data_loader) + batch_idx)
        return total_loss

    def test(self, data_loader):
        self.model.eval()
        test_loss = 0
        correct = 0
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = self.model(data)

            # sum up batch loss
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(data_loader.dataset)
        accuracy = 100. * correct / len(data_loader.dataset)
        # print('\nTest set {} : Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
        #       .format(test_set, test_loss, correct, len(data_loader.dataset),
        #               accuracy))
        return accuracy, test_loss

    def train_model(self, num_epoch=100, train_dir='./data/', test_dir='./test1/', save_path='./saved_models'):
        save_path = os.path.join(save_path, str(datetime.now()).split('.')[0])
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        print('Saving models at ', save_path)
        train_dataloader = self.make_dataloader(data_dir=train_dir)
        test_dataloader = self.make_dataloader(data_dir=test_dir)
        for epoch in range(1, num_epoch + 1):
            training_loss = self.train_epoch(train_dataloader, epoch)
            # break
            test_accuracy, test_loss = self.test(test_dataloader)
            print('Test accuracy at epoch : ', epoch, ' is ', test_accuracy, 'and loss is ', test_loss)
            self.validation_acc.add_scalar('Acc/Val', test_accuracy, epoch)
            self.validation_acc.add_scalar('Loss/Val', test_loss, epoch)
            if epoch % 100 == 0:
                self.save_model(epoch, training_loss, save_path)
        


if __name__ == "__main__":
    classifier = DigitClassifier()
    classifier.train_model(train_dir='./prepared_data')
    # test_dir = './test1/'
    # test_dataloader = classifier.make_dataloader(test_dir)
    # checkpoint_path = './saved_models/2019-10-11 13:16:18/ep1.pth.tar'
    # model = Model()
    # model = load_model(model, checkpoint_path)
    # accuracy, pred, output=test_model(model, test_dataloader)
    # print(accuracy)
    # print(pred)
    # print(output)
