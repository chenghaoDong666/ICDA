import os
import torch
import torch.utils.data
import torch.distributed
import torch.backends.cudnn


class Prototype():
    def __init__(self, feature_num, class_num, momentum, use_momentum):
        super(Prototype, self).__init__()
        self.class_num = class_num
        self.feature_num = feature_num
        # momentum 
        self.use_momentum = use_momentum
        self.momentum = momentum

        # init prototype
        self.Proto = torch.zeros(self.class_num, feature_num)
        #self.Amount = torch.zeros(self.class_num).to(device)

    def update(self, features, labels):
        mask = (labels != 255)
        # remove IGNORE_LABEL pixels
        # (N,)
        labels = labels[mask]
        # (N,A)
        features = features[mask]
        # 不用momentum也用了移动平均,但是不是固定的值,而是实时计算出来的权重
        if not self.use_momentum:
            N, A = features.size()
            C = self.class_num
            # refer to SDCA for fast implementation
            features = features.view(N, 1, A).expand(N, C, A)
            onehot = torch.zeros(N, C).cuda()
            onehot.scatter_(1, labels.view(-1, 1), 1)
            NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A)
            features_by_sort = features.mul(NxCxA_onehot)
            Amount_CXA = NxCxA_onehot.sum(0)
            Amount_CXA[Amount_CXA == 0] = 1
            mean = features_by_sort.sum(0) / Amount_CXA

            sum_weight = onehot.sum(0).view(C, 1).expand(C, A)
            weight = sum_weight.div(
                sum_weight + self.Amount.view(C, 1).expand(C, A)
            )
            weight[sum_weight == 0] = 0
            self.Proto = (self.Proto.mul(1 - weight) + mean.mul(weight)).detach()
            self.Amount = self.Amount + onehot.sum(0)
        else:
            # momentum implementation
            ids_unique = labels.unique()
            for i in ids_unique:
                i = i.item()
                mask_i = (labels == i)
                feature = features[mask_i]
                feature = torch.mean(feature, dim=0)
                # self.Amount[i] += len(mask_i)
                self.Proto[i, :] = self.momentum * feature + self.Proto[i, :] * (1 - self.momentum)
            #print(self.Proto)
            
        
"""     def save(self, name):
        torch.save({'Proto': self.Proto.cpu(),
                    'Amount': self.Amount.cpu()
                    },
                   os.path.join(self.cfg.OUTPUT_DIR, name)) """
