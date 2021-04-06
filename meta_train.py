from arguments import parse_args
import configs
from dataloader.datamgr import SetDataManager, SimpleDataManager
from resnet18.backbone import EmbeddingNet
import os
import torch
from vit.crosstransformers import CrossTransformer

def get_features_res18(params, model, data, feat_tensor= False):
    with torch.no_grad():
        n_way, x, height, width, channel = data[0].size()
        for i in range(data[1].shape[0]):
            data[1][i, :] = i

        support_xs, query_xs = data[0][:,0:int(x/2),:,:,:].to(params.device), data[0][:,int(x/2):,:,:,:].to(params.device)

        #support_xs = data[0]
        support_xs = support_xs.contiguous().view(-1, height, width, channel).to(params.device)
        query_xs = query_xs.contiguous().view(-1, height, width, channel).to(params.device)
        support_ys, query_ys = data[1][:,0:int(x/2)].to(params.device), data[1][:,int(x/2):].to(params.device)
        support_ys, query_ys = support_ys.contiguous().view(-1).to(params.device), query_ys.contiguous().view(-1).to(params.device)
        support_features, query_features = model(support_xs, feat_tensor=feat_tensor), model(query_xs, feat_tensor=feat_tensor)
    return support_features, support_ys, query_features, query_ys

def get_xs_ready(params, data):
    with torch.no_grad():
        n_way, x, height, width, channel = data[0].size()
        for i in range(data[1].shape[0]):
            data[1][i, :] = i

        support_xs, query_xs = data[0][:,0:int(x/2),:,:,:].to(params.device), data[0][:,int(x/2):,:,:,:].to(params.device)

        #support_xs = data[0]
        #support_xs = support_xs.contiguous().view(-1, height, width, channel).to(params.device)
        print(support_xs.shape)
        query_xs = query_xs.contiguous().view(-1, height, width, channel).to(params.device)
        support_ys, query_ys = data[1][:,0:int(x/2)].to(params.device), data[1][:,int(x/2):].to(params.device)
        support_ys, query_ys = support_ys.contiguous().view(-1).to(params.device), query_ys.contiguous().view(-1).to(params.device)

    return support_xs, support_ys, query_xs, query_ys

def meta_train_res(args, model, cross_transformer, datamgr):
    #optimizer = torch.optim.Adam([{'params': vae.parameters()}, {'params': cos_scores.parameters(), 'lr': 1e-2}], lr=1e-3)
    optimizer = torch.optim.Adam(cross_transformer.parameters(), lr = 1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    #CE = nn.CrossEntropyLoss()
    for j in range(100):
        data_loader = datamgr.get_data_loader(args.loadfile, aug=False)
        for idx, data in enumerate(data_loader):
            support_xs, support_ys, query_features, query_ys = get_xs_ready(args, data)

            no_per_class = int(support_ys.shape[0]/args.n_ways)
            optimizer.zero_grad()
            # generate from outside so that gradients propagate
            # Train
            total_loss = loss #+ loss_triplet +  loss_gen
            total_loss.backward()
            optimizer.step()
            if idx%200 ==0:
                print('Iteration: ', j, "Episode: ", idx," Loss: ", total_loss.item(), 'triplet_loss: ', loss_triplet.item(), " vae_loss: ", loss.item())# 'learning rate: ', scheduler.get_lr())
        scheduler.step()


if __name__ == '__main__':
    params = parse_args('test')
    image_size = 224
    datamgr = SetDataManager(params, image_size)
    split = 'base'
    params.loadfile = configs.data_dir[params.dataset] + split + '.json'
    #data_loader = datamgr.get_data_loader(loadfile, aug=False)
    if params.dataset == 'CUB':
        params.num_classes = 100
    elif params.dataset == 'tieredImagenet':
        params.num_classes = 351
    else:
        params.num_classes = 64
    if params.model =='resnet18':
        params.file_path = '%s/checkpoints/CUB_rfs.tar' %(os.path.dirname(os.path.abspath(__file__)))
        model = EmbeddingNet(params, dense=False)
    print(params.file_path)
    print(os.path.dirname(os.path.abspath(__file__)))
    cross_transformer = CrossTransformer()

    meta_train_res(params, model, cross_transformer, datamgr)