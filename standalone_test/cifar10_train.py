'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

import os
import argparse

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
plus_epoch = 50 # total epoch == start_epoch + plus_epoch
print("Training with ... ", device)

# CIFAR10 dataset prepare
print('==> Preparing data CIFAR10..')
best_prec = 0.0

transform_train = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Should be an abs path here
trainset = torchvision.datasets.CIFAR10(root='../../dataset', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='../../dataset', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# Model
print('==> Building model.. vgg16')
# net = VGG('VGG19')
# net = ResNet18()
# net = models.resnet18()
net = models.vgg16()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('../checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        print('Training[{}/{}] Loss: {:.2f} | Acc: {:.2f}%'.format(batch_idx, len(trainloader), train_loss/(batch_idx+1), 100.*correct/total))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            print('Testing[{}/{}] Loss: {:.2f} | Acc: {:.2f}%'.format(batch_idx, len(testloader), test_loss/(batch_idx+1), 100.*correct/total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('../checkpoint'):
            os.mkdir('../checkpoint')
        torch.save(state, '../checkpoint/ckpt.pth')
        torch.save(net, '../checkpoint/{}_ckpt.pth'.format("vgg16"))
        torch.save(net.state_dict(), '../checkpoint/{}_stat.pth'.format("vgg16"))

        # Save jit script for c/c++
        # Using torch.jit.trace()
        # dummy_input = torch.rand(64, 3, 224, 224).to(device)
        # trace_model = torch.jit.trace(net, dummy_input)
        # torch.jit.save(trace_model,'./jit/resnet_trace.pt')
        # Using torch.jit.script()
        #  script_model = torch.jit.script(net.cpu())
        #  torch.jit.save(script_model, './jit/resnet_script.pt')
        best_acc = acc


for epoch in range(start_epoch, start_epoch+plus_epoch):
# for epoch in range(2):
    train(epoch)
    test(epoch)
    # scheduler.step()
