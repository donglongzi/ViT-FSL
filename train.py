import torch
import torch.nn as nn
import os
import numpy as np
from dataloader.datamgr import SimpleDataManager
import configs
from arguments import parse_args
#from resnet18.backbone import ClassificationNetwork
from vit.vit_model import ViT

def adjust_learning_rate(params, epoch, optimizer):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    #print(params.lr_decay_epochs)
    steps = np.sum(epoch > np.asarray(params.lr_decay_epochs))
    if steps > 0:
        new_lr = params.lr * (0.1 ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

def train(base_loader, val_loader, model, start_epoch, stop_epoch, params):
    # optimizer from rfs
    optimizer = torch.optim.SGD(model.parameters(), lr=params.lr, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    max_acc = 0
    model.to(params.device)

    for epoch in range(start_epoch,stop_epoch):
        adjust_learning_rate(params, epoch, optimizer)
        model.train()
        train_loop(params, model, epoch, base_loader,  optimizer, criterion)
        #scheduler.step()
        model.eval()

        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)

        #if epoch == stop_epoch-1:
    outfile = os.path.join(params.checkpoint_dir, 'vit_rfs.tar')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
            #    'loss': loss
    }, outfile)

def train_loop(params, model, epoch, train_loader, optimizer, criterion):
    print_freq = 100
    avg_loss = 0

    for i, (x, y) in enumerate(train_loader):
        #-----------Problem in training for CUB database
        #getting from 0-198 labels, converting them to 0-99
        if params.dataset == "CUB":
            y = y/2
        x, y = x.to(params.device), y.long().to(params.device)
        outputs = model(x, feat= False)
        #print(outputs.shape)
        loss = criterion(outputs, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss = avg_loss + loss.item()

        if i % print_freq == 0:
            # print(optimizer.state_dict()['param_groups'][0]['lr'])
            print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader), avg_loss / float(i + 1)))


if __name__ == '__main__':
    np.random.seed(10)
    params = parse_args('train')
    image_size = 224
    base_file = configs.data_dir[params.dataset] + 'base.json'
    val_file = configs.data_dir[params.dataset] + 'val.json'
    params.checkpoint_dir = '%s/checkpoints' %(os.path.dirname(os.path.abspath(__file__)))
    print(params.checkpoint_dir)
    print(os.path.dirname(os.path.abspath(__file__)))
    iterations = params.lr_decay_epochs.split(',')
    params.lr_decay_epochs = list([])
    for it in iterations:
        params.lr_decay_epochs.append(int(it))

    base_datamgr = SimpleDataManager(image_size, batch_size=params.batch_size)
    base_loader = base_datamgr.get_data_loader(base_file, aug=True)
    val_datamgr = SimpleDataManager(image_size, batch_size=params.test_batch_size)
    val_loader = val_datamgr.get_data_loader(val_file, aug=False)
    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch

    if params.dataset == 'CUB':
        params.num_classes = 100
    elif params.dataset == 'tieredImagenet':
        params.num_classes = 351
    else:
        params.num_classes = 64
    model = ViT(
        image_size=image_size,
        patch_size=16,
        num_classes=params.num_classes,
        dim=512,
        depth=4,
        heads=16,
        mlp_dim=1024,
        dropout=0.1,
        emb_dropout=0.1
    ).to(params.device)
    train(base_loader, val_loader, model, start_epoch, stop_epoch, params)




