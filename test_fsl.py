import torch
from dataloader.datamgr import SetDataManager
import configs
import os
from arguments import parse_args
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import scipy as sp
from scipy.stats import t
from resnet18.backbone import EmbeddingNet
from sklearn import metrics
from tqdm import tqdm
from vit.vit_model import ViT


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), sp.stats.sem(a)
    h = se * t._ppf((1+confidence)/2., n-1)
    return m, h

def convert_to_few_shot_labels(data):
    for i in range(data.shape[0]):
        data[i,:] = i
    return data

def get_predictions(params, model, data):
    with torch.no_grad():
        data[1] = convert_to_few_shot_labels(data[1])
        n_way, _, height, width, channel = data[0].size()

        support_xs, query_xs = data[0][:, :params.n_shots], data[0][:, params.n_shots:]
        support_ys, query_ys = data[1][:, :params.n_shots], data[1][:, params.n_shots:]
        support_xs = support_xs.contiguous().view(-1, height, width, channel).to(params.device)
        query_xs = query_xs.contiguous().view(-1, height, width, channel).to(params.device)

        support_ys = support_ys.contiguous().view(-1)
        query_ys = query_ys.contiguous().view(-1)
        if params.model =='vit_pre':
            support_features, query_features = model.forward_features(support_xs), model.forward_features(query_xs)
        elif params.model =='bit_pre':
            support_features, query_features = model.forward_features(support_xs), model.forward_features(query_xs)
            support_features, query_features = model.head.global_pool(support_features).view(support_features.size(0),-1), model.head.global_pool(query_features).view(query_features.size(0), -1)
        else:
            support_features, query_features = model(support_xs), model(query_xs)


    support_features, query_features = support_features.detach().cpu(), query_features.detach().cpu()
    #for i in range(support_features.shape[1]):
     #   print(support_features[0][i].item())

    support_ys, query_ys = support_ys.detach().cpu(), query_ys.detach().cpu()
    if params.classifier == 'LR':
        clf = LogisticRegression(penalty='l2', random_state=0, C=1.0, solver='lbfgs', max_iter=1000, multi_class='multinomial')
        clf.fit(support_features, support_ys)
        query_ys_pred = clf.predict(query_features)
    elif params.classifier == 'SVM':
        clf = SVC(C=10, gamma='auto', kernel='linear', probability=True)
        clf.fit(support_features, support_ys)
        query_ys_pred = clf.predict(query_features)
    return query_ys, query_ys_pred

def few_shot_test(params, model, testloader):
    model = model.eval()
    model.to(params.device)
    acc = []
    for idx, data in tqdm(enumerate(testloader)):
        query_ys, query_ys_pred = get_predictions(params, model, data)
        acc.append(metrics.accuracy_score(query_ys, query_ys_pred))
    return mean_confidence_interval(acc)

if __name__ == '__main__':
    params = parse_args('test')
    image_size = 224
    datamgr = SetDataManager(params, image_size)
    split = 'novel'
    loadfile = configs.data_dir[params.dataset] + split + '.json'
    data_loader = datamgr.get_data_loader(loadfile, aug=False)
    if params.dataset == 'CUB':
        params.num_classes = 100
    elif params.dataset == 'tieredImagenet':
        params.num_classes = 351
    else:
        params.num_classes = 64
    if params.model =='resnet18':
        params.file_path = '%s/checkpoints/CUB_rfs.tar' %(os.path.dirname(os.path.abspath(__file__)))
        model = EmbeddingNet(params, dense=False)
    elif params.model =='vit':
        params.file_path = '%s/checkpoints/vit_rfs.tar' %(os.path.dirname(os.path.abspath(__file__)))
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
        )#.to(params.device)
        ckpt = torch.load(params.file_path)
        model.load_state_dict(ckpt['model_state_dict'])
    elif params.model =='vit_pre':
        import timm
        model = timm.create_model('vit_base_patch16_224_in21k', num_classes=params.num_classes, pretrained=True)
        avail_pretrained_models = timm.list_models('*vit*',pretrained=True)
    elif params.model == 'bit_pre':
        import timm
        model = timm.create_model('resnetv2_152x2_bitm', num_classes=1000, pretrained=True)
        #avail_pretrained_models = timm.list_models('*vit*', pretrained=True)
        #print(avail_pretrained_models)
    #print(params.file_path)
    #print(os.path.dirname(os.path.abspath(__file__)))


    test_acc, test_std = few_shot_test(params, model, data_loader)
    print("model: {}, Classifier: {}, shots: {}, dataset: {}, acc: {:0.2f} +- {:0.2f}".format(params.model, params.classifier, params.n_shots, params.dataset, test_acc*100, test_std*100))


