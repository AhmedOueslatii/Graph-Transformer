import sys
import os
import torch
import random
import numpy as np

from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .ViT import *
from .gcn import GCNBlock

from torch_geometric.nn import GCNConv, DenseGraphConv, dense_mincut_pool
from torch.nn import Linear
class Classifier(nn.Module):
    def __init__(self, n_class):
        super(Classifier, self).__init__()
        self.embed_dim = 64
        self.num_layers = 5
        self.node_cluster_num = 100

        self.transformer = VisionTransformer(num_classes=n_class, embed_dim=self.embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.criterion = nn.CrossEntropyLoss()

        self.conv1 = GCNBlock(512,self.embed_dim,1,1,1,0.,0)       # Simplified GCNBlock

        self.bn1 = nn.BatchNorm1d(self.embed_dim)  # Batch normalization after GCNBlock
        self.dropout1 = nn.Dropout(p=0.5)
        self.pool1 = Linear(self.embed_dim, self.node_cluster_num)     
                                             # 100-> 20


    def forward(self,node_feat,labels,adj,mask,is_print=False, graphcam_flag=False):
        # node_feat, labels = self.PrepareFeatureLabel(batch_graph)
        cls_loss=node_feat.new_zeros(self.num_layers)
        rank_loss=node_feat.new_zeros(self.num_layers-1)
        X=node_feat
        p_t=[]
        pred_logits=0
        visualize_tools=[]
        visualize_tools1=[labels.cpu()]
        embeds=0
        concats=[]
        
        layer_acc=[]
                
        X=mask.unsqueeze(2)*X
        X = self.conv1(X, adj, mask)
        X = self.bn1(X.permute(0, 2, 1)).permute(0, 2, 1)  # Apply batch normalization
        X = self.dropout1(X) 
        s = self.pool1(X)

        if graphcam_flag:
            s_matrix = torch.argmax(s[0], dim=1)
            from os import path
            torch.save(s_matrix, 'graphcam/s_matrix.pt')
            torch.save(s[0], 'graphcam/s_matrix_ori.pt')
            
            if path.exists('graphcam/att_1.pt'):
                os.remove('graphcam/att_1.pt')
                os.remove('graphcam/att_2.pt')
                os.remove('graphcam/att_3.pt')
    
        X, adj, mc1, o1 = dense_mincut_pool(X, adj, s, mask)
        b, _, _ = X.shape
        cls_token = self.cls_token.repeat(b, 1, 1)
        X = torch.cat([cls_token, X], dim=1)

        out = self.transformer(X)

        # loss
        loss = self.criterion(out, labels)
        loss = loss + mc1 + o1
        # pred
        pred = out.data.max(1)[1]

        if graphcam_flag:
            print('GraphCAM enabled')
            p = F.softmax(out)
            torch.save(p, 'graphcam/prob.pt')
            index = np.argmax(out.cpu().data.numpy(), axis=-1)

            for index_ in range(3):
                one_hot = np.zeros((1, out.size()[-1]), dtype=np.float32)
                one_hot[0, index_] = out[0][index_]
                one_hot_vector = one_hot
                one_hot = torch.from_numpy(one_hot).requires_grad_(True)
                one_hot = torch.sum(one_hot.cuda() * out)       #!!!!!!!!!!!!!!!!!!!!out-->p
                self.transformer.zero_grad()
                one_hot.backward(retain_graph=True)

                kwargs = {"alpha": 1}
                cam = self.transformer.relprop(torch.tensor(one_hot_vector).to(X.device), method="transformer_attribution", is_ablation=False, 
                                            start_layer=0, **kwargs)

                torch.save(cam, 'graphcam/cam_{}.pt'.format(index_))

        return pred,labels,loss
