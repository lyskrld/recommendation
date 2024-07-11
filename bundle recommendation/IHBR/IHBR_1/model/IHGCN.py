import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np
from .model_base import Info, Model
from config import CONFIG
import torch_sparse
from torch_sparse import SparseTensor
from torch_sparse.mul import mul
from torch.nn.parameter import  Parameter
import pdb
import att_layer
from dataset import MyDataset
from lightfm import LightFM
from torch.autograd import Variable
from .disentangle_module import Disentangle
from Attlayerself import AttLayerSelf
from Attlayer import AttLayer
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
def graph_generating(raw_graph, row, col):
    if raw_graph.shape == (row, col):
        graph = sp.bmat([[sp.identity(raw_graph.shape[0]), raw_graph],
                             [raw_graph.T, sp.identity(raw_graph.shape[1])]])
    else:
        raise ValueError(r"raw_graph's shape is wrong")
    return graph

def laplace_transform(graph):
    rowsum_sqrt = sp.diags(1/(np.sqrt(graph.sum(axis=1).A.ravel()) + 1e-8))
    colsum_sqrt = sp.diags(1/(np.sqrt(graph.sum(axis=0).A.ravel()) + 1e-8))
    graph = rowsum_sqrt @ graph @ colsum_sqrt
    return graph

def to_tensor(graph):
    graph = graph.tocoo()
    values = graph.data
    indices = np.vstack((graph.row, graph.col))
    graph = torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(values), 
                                          torch.Size(graph.shape))
    return graph


class GaussianEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super(GaussianEmbedding, self).__init__()
        self.mean_embedding = nn.Linear(embedding_dim, embedding_dim)
        self.var_embedding = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        mean = self.mean_embedding(x)
        var = self.var_embedding(x)
        var = F.elu(var) + 1.0
        epsilon = torch.randn_like(var)
        sample = mean + torch.sqrt(var) * epsilon
        return sample

class IHGCN_Info(Info):
    def __init__(self, embedding_size, embed_L2_norm, mess_dropout, node_dropout, num_layers, act=nn.LeakyReLU()):
        super().__init__(embedding_size, embed_L2_norm)
        self.act = act
        assert 1 > mess_dropout >= 0
        self.mess_dropout = mess_dropout
        assert 1 > node_dropout >= 0
        self.node_dropout = node_dropout
        assert isinstance(num_layers, int) and num_layers > 0
        self.num_layers = num_layers


class IHGCN(Model):

   def get_infotype(self):
        return IHGCN_Info
   
   def __init__(self, info, dataset, raw_graph, device, pretrain=None):
        super().__init__(info, dataset, create_embeddings=True)

        self.n_factors = 2
        self.n_layers =1
        self.n_iterations = 2
        self.epison = 1e-8
        self.act = self.info.act
        self.act=nn.LeakyReLU()
        self.num_layers = self.info.num_layers
        self.device = device
        #Youshu  0.25
        #Netease 0.2
        self.c_temp = 0.25
        self.disentangle=Disentangle()

         #  Dropouts
        self.mess_dropout = nn.Dropout(self.info.mess_dropout, True)
        self.node_dropout = nn.Dropout(self.info.node_dropout, True)


        

        # self.gating_weightub=nn.Parameter(
        #     torch.FloatTensor(1,self.embedding_size))
        # nn.init.xavier_normal_(self.gating_weightub.data)
        # self.gating_weightu=nn.Parameter( 
        #     torch.FloatTensor(self.embedding_size,self.embedding_size))
        # nn.init.xavier_normal_(self.gating_weightu.data)

        # self.gating_weightib=nn.Parameter( 
        #     torch.FloatTensor(1,self.embedding_size))
        # nn.init.xavier_normal_(self.gating_weightib.data)
        # self.gating_weighti=nn.Parameter(
        #     torch.FloatTensor(self.embedding_size,self.embedding_size))
        # nn.init.xavier_normal_(self.gating_weighti.data)


        # self.gating_weightbb=nn.Parameter( 
        #     torch.FloatTensor(1,self.embedding_size))
        # nn.init.xavier_normal_(self.gating_weightbb.data)
        # self.gating_weightb=nn.Parameter(
        #     torch.FloatTensor(self.embedding_size,self.embedding_size))
        # nn.init.xavier_normal_(self.gating_weighti.data)


        # 创建GaussianEmbedding实例
        # embedding = GaussianEmbedding(self.embedding_size)

        self.users_feature = nn.Parameter(torch.FloatTensor(self.num_users, self.embedding_size))
        nn.init.xavier_normal_(self.users_feature)

        # input_data_u = torch.randn(self.num_users, self.embedding_size)
        # self.users_feature = nn.Parameter(embedding(input_data_u))
        

        self.bundles_feature = nn.Parameter(torch.FloatTensor(self.num_bundles, self.embedding_size))
        nn.init.xavier_normal_(self.bundles_feature)

        # input_data_b = torch.randn(self.num_bundles, self.embedding_size)
        # self.bundles_feature = nn.Parameter(embedding(input_data_b))
        

        self.items_feature = nn.Parameter(torch.FloatTensor(self.num_items, self.embedding_size))
        nn.init.xavier_normal_(self.items_feature)

        # input_data_i = torch.randn(self.num_items, self.embedding_size)
        # self.items_feature = nn.Parameter(embedding(input_data_i))
       

        # users_feature,items_feature = self.pre_item_user()
        # #self.users_feature=nn.Parameter(torch.tensor(users_feature))
        # self.items_feature=nn.Parameter(torch.tensor(items_feature))
        # nn.init.xavier_normal_(self.users_feature)
        # nn.init.xavier_normal_(self.items_feature)

        # bundles_feature=self.pre_bundle(self.items_feature,self.users_feature,self.bundles_feature)
        # self.bundles_feature=nn.Parameter(bundles_feature)
        # nn.init.xavier_normal_(self.bundles_feature)

        # users_feature,item_feature=self.load_embeddings_from_path()
        # self.users_feature=nn.Parameter(torch.tensor(users_feature))
        # self.items_feature=nn.Parameter(torch.tensor(item_feature))
       
        # bundles_feature=self.pre_bundle(self.items_feature,self.users_feature,self.bundles_feature)
        # self.bundles_feature=nn.Parameter(bundles_feature)
        # nn.init.xavier_normal_(self.bundles_feature)

        

        assert isinstance(raw_graph, list)
        ub_graph, ui_graph, bi_graph = raw_graph
        
        ui_graph_coo, ub_graph_coo, bi_graph_coo = ui_graph.tocoo(), ub_graph.tocoo(), bi_graph.tocoo()
        ub_indices = torch.tensor([ub_graph_coo.row, ub_graph_coo.col], dtype=torch.long)
        ub_values = torch.ones(ub_graph_coo.data.shape, dtype=torch.float)

        bi_indices = torch.tensor([bi_graph_coo.row, bi_graph_coo.col], dtype=torch.long)
        bi_values = torch.ones(bi_graph_coo.data.shape, dtype=torch.float)

        ui_e_indices, ui_e_values = torch_sparse.spspmm(ub_indices, ub_values, bi_indices, bi_values, self.num_users,
                                                        self.num_bundles, self.num_items)

        ui_graph_e = sp.csr_matrix((np.array([1] * len(ui_e_values)), (ui_e_indices[0].numpy(), ui_e_indices[1].numpy())),
            shape=(self.num_users, self.num_items))
        ui_graph_e_coo = ui_graph_e.tocoo()
        ui_graph_e_coo = ui_graph_e.tocoo()
        self.ui_mask = ui_graph_e[ui_graph_coo.row, ui_graph_coo.col]
        self.ui_e_mask = ui_graph[ui_graph_e_coo.row, ui_graph_e_coo.col]
        self.bi_graph, self.ui_graph = bi_graph, ui_graph
        if ui_graph.shape == (self.num_users, self.num_items):
            # add self-loop
            atom_graph = sp.bmat([[sp.identity(ui_graph.shape[0]), ui_graph],
                                  [ui_graph.T, sp.identity(ui_graph.shape[1])]])
        else:
            raise ValueError(r"raw_graph's shape is wrong")
        
        self.ui_atom_graph = to_tensor(laplace_transform(atom_graph)).to(device)
        if bi_graph.shape == (self.num_bundles, self.num_items):
            # add self-loop
            atom_graph = sp.bmat([[sp.identity(bi_graph.shape[0]), bi_graph],
                                  [bi_graph.T, sp.identity(bi_graph.shape[1])]])
        else:
            raise ValueError(r"raw_graph's shape is wrong")
        
        self.bi_atom_graph = to_tensor(laplace_transform(atom_graph)).to(device)
       
        if bi_graph.shape == (self.num_bundles, self.num_items):
            tmp = bi_graph.tocoo()
            self.bi_graph_h = list(tmp.row)
            self.bi_graph_t = list(tmp.col)
            self.bi_graph_shape = bi_graph.shape
        else:
            raise ValueError(r"raw_graph's shape is wrong")
        # self.bi_graph = to_tensor(laplace_transform(bi_graph)).to(device)

        if ui_graph.shape == (self.num_users, self.num_items):
            # add self-loop
            tmp = ui_graph.tocoo()
            self.ui_graph_v = torch.tensor(tmp.data, dtype=torch.float).to(device)
            self.ui_graph_h = list(tmp.row)
            self.ui_graph_t = list(tmp.col)
            self.ui_graph_shape = ui_graph.shape
        else:
            raise ValueError(r"raw_graph's shape is wrong")
        if ub_graph.shape == (self.num_users, self.num_bundles):
            # add self-loop
            tmp = ub_graph.tocoo()
            self.ub_graph_v = torch.tensor(tmp.data, dtype=torch.float).to(device)
            self.ub_graph_h = list(tmp.row)
            self.ub_graph_t = list(tmp.col)
            self.ub_graph_shape = ub_graph.shape
        else:
            raise ValueError(r"raw_graph's shape is wrong")
        # self.ui_graph = to_tensor(laplace_transform(ui_graph)).to(device)
        print('finish generating bi, ui graph')

        #  deal with weights
        bi_norm = sp.diags(1 / (np.sqrt((bi_graph.multiply(bi_graph)).sum(axis=1).A.ravel()) + self.epison)) @ bi_graph
        bb_graph = bi_norm @ bi_norm.T

        bundle_size = bi_graph.sum(axis=1) + self.epison
        bi_graph = sp.diags(1 / bundle_size.A.ravel()) @ bi_graph
        
        self.pooling_graph = to_tensor(bi_graph).to(device)
        
        if ub_graph.shape == (self.num_users, self.num_bundles) \
                and bb_graph.shape == (self.num_bundles, self.num_bundles):
            # add self-loop
            non_atom_graph = sp.bmat([[sp.identity(ub_graph.shape[0]), ub_graph],
                                      [ub_graph.T, bb_graph]])
        else:
            raise ValueError(r"raw_graph's shape is wrong")
        self.non_atom_graph = to_tensor(laplace_transform(non_atom_graph)).to(device)
        print('finish generating non-atom graph')

        
        # Layers
        self.dnns_atom = nn.ModuleList([nn.Linear(
            self.embedding_size*(l+1), self.embedding_size) for l in range(self.n_layers)])
        self.dnns_non_atom = nn.ModuleList([nn.Linear(
            self.embedding_size*(l+1), self.embedding_size) for l in range(self.n_layers)])
        
        self.att= nn.ModuleList([att_layer.Att(self.embedding_size * (l + 1)) for l in range(self.n_layers)])
        self.adapter=nn.ModuleList([att_layer.Adapter(self.embedding_size * (l + 1)) for l in range(self.n_layers)])
        self.adapter1=att_layer.Adapter(384)
        self.adapter2=att_layer.Adapter(384)#384
        #self.matatt = att_layer.MHAtt(8,self.embedding_size,self.embedding_size,self.embedding_size)
        self.matatt=att_layer.MHAtt(8,192,192,192)
        # self.fc1 = nn.Linear(192,32770)
        # self.fc2 = nn.Linear(32770,192)
        # self.loss_KLD = nn.KLDivLoss()
        # self.loss_fn = nn.CrossEntropyLoss()
        # self.attself=AttLayerSelf(1e-6)
        # self.attlayer=AttLayer(192,0)

#    def self_gatingu(self,em):
#         return torch.multiply(em, torch.sigmoid(torch.matmul(em,self.gating_weightu) + self.gating_weightub))
#    def self_gatingi(self,em):
#         return torch.multiply(em, torch.sigmoid(torch.matmul(em,self.gating_weighti) + self.gating_weightib))
#    def self_gatingb(self,em):
#         return torch.multiply(em, torch.sigmoid(torch.matmul(em,self.gating_weightb) + self.gating_weightbb))
#预处理
#    def load_embeddings_from_path(self):
#         with open("/data/lzm/IHBR - 副本/bpr_user_avg_items_Youshu_2023-08-01-2056/model.pkl", 'rb') as pkl:
#                 bpr_model = pickle.load(pkl)
#                 bpr_users = torch.tensor(bpr_model.user_factors.to_numpy()) / torch.tensor(bpr_model.user_norms.to_numpy()).reshape(-1, 1)
#                 bpr_items= torch.tensor(bpr_model.item_factors.to_numpy()) / torch.tensor(bpr_model.item_norms.to_numpy()).reshape(-1, 1)
                
#         return bpr_users,bpr_items
   
#    def pre_item_user(self):
#         """ Generates bundle recommendations for each user in the dataset """
#         # Create a model from the input data
#         """ model = BayesianPersonalizedRanking(factors=args.size - 1, verify_negative_samples=True, iterations=100)
#         dataset = Dataset(path='Data', args=args, use_mini_test=False)
#         train_interactions = dataset.ground_truth_u_i if args.avg_items else dataset.ground_truth_u_b_train
#         model.fit(train_interactions, show_progress=True) """
        
        
#         model = LightFM(loss='bpr', no_components=self.embedding_size) 
#         dataset = MyDataset(path='data', use_mini_test=False)
#         train_interactions = dataset.ground_truth_u_i 
#         model.fit(train_interactions, epochs=100,verbose=True)
#         #train_precision = auc_score(model, train_interactions).mean()
#         #print('Precision: train %.2f' % train_precision)
#         user_embeddings = model.user_embeddings
#         item_embeddings = model.item_embeddings
        
#         return user_embeddings,item_embeddings


#    def pre_bundle(self,item_embeddings):
#         be = []
#         w = []
#         dataset = MyDataset(path='data', use_mini_test=False)
#         for bundle in range(dataset.num_bundles):
#             items_in_bundle = dataset.ground_truth_b_i[bundle].nonzero()[1]
#             bi = item_embeddings[items_in_bundle]
#             def create_custom_forward(module):
#                 def custom_forward(*inputs):
#                     return module(*inputs)

#                 return custom_forward

#             bi_attself = torch.utils.checkpoint.checkpoint(
#                 create_custom_forward(self.attself), bi)
#             b = torch.utils.checkpoint.checkpoint(
#                 create_custom_forward(self.attlayer), bi_attself)
#             # bi_attself=self.attself(bi)
#             # b=self.attlayer(bi_attself)
#             be.append(b)
#         be = torch.stack(be)
#         return be
 
  
   
   

   def interpolation_data_augmentation(self,embedding_a, embedding_b, alpha=10):
        """
        插值数据扩充方法，对两个域中用户嵌入进行线性插值

        参数:
            embedding_a (Tensor): 域 A 中的用户嵌入
            embedding_b (Tensor): 域 B 中的用户嵌入
            alpha (float): 插值参数,控制插值程度(默认为1.0)

        返回:
            augmented_embedding (Tensor): 扩充后的用户嵌入
        """
        lambda_ = torch.distributions.beta.Beta(alpha, alpha).sample().item()
        augmented_embedding = lambda_ * embedding_a + (1 - lambda_) * embedding_b

        

        return augmented_embedding
   #Youshu 1,10
   #Netease 10,10
   def RepMixup(self,embedding_a, embedding_b, alpha=10):
        """
        插值数据扩充方法，对两个域中用户嵌入进行线性插值

        参数:
            embedding_a (Tensor): 域 A 中的用户嵌入
            embedding_b (Tensor): 域 B 中的用户嵌入
            alpha (float): 插值参数,控制插值程度(默认为1.0)

        返回:
            augmented_embedding (Tensor): 扩充后的用户嵌入
        """
        lambda_ = torch.distributions.beta.Beta(alpha, alpha).sample().item()
        augmented_embedding = lambda_ * embedding_a*embedding_b + (1 - lambda_) * self.interpolation_data_augmentation(embedding_a, embedding_b)

        

        return augmented_embedding



   def one_propagate(self, graph, A_feature, B_feature, dnns):
        # node dropout on graph
        indices = graph._indices()
        values = graph._values()
        values = self.node_dropout(values)
        graph = torch.sparse.FloatTensor(
            indices, values, size=graph.shape)

        # propagate
        features = torch.cat((A_feature, B_feature), 0)
        all_features = [features]
        for i in range(self.n_layers):
            features = self.mess_dropout(torch.cat([self.act(
                dnns[i](torch.matmul(graph,self.adapter[i](features)))), features], 1))
                # dnns[i](torch.matmul(graph,features))), features], 1))
            
            all_features.append(F.normalize(features)) 

        all_features = torch.cat(all_features, 1)
        A_feature, B_feature = torch.split(
            all_features, (A_feature.shape[0], B_feature.shape[0]), 0)
        return A_feature, B_feature,all_features
   
#    def UI_propagate(self, graph, A_feature, B_feature, mess_dropout):
#         features = torch.cat((A_feature, B_feature), 0)
#         all_features = [features]

#         for i in range(self.num_layers):
#             features = torch.spmm(graph, features)
            
#             features = mess_dropout(features)

#             features = features / (i+2)
#             all_features.append(F.normalize(features, p=2, dim=1))

#         all_features = torch.stack(all_features, 1)
#         all_features = torch.sum(all_features, dim=1).squeeze(1)

#         A_feature, B_feature = torch.split(all_features, (A_feature.shape[0], B_feature.shape[0]), 0)

#         return A_feature, B_feature
    
   def propagate(self):
        #  =============================  item level propagation  =============================
        #items= self.matatt(self.items_feature, self.items_feature, self.items_feature)
        #users_feature_gate=self.self_gatingu(self.users_feature)
        #items_feature_gate=self.self_gatingi(self.items_feature)
        #bundles_feature_gate=self.self_gatingb(self.bundles_feature)

    
        atom_users_feature, atom_items_feature,u_i = self.one_propagate(
            self.ui_atom_graph,self.users_feature, self.items_feature,self.dnns_atom)
       
        #  ============================= bundle level propagation =============================
        #cur_ub=torch.cat((self.users_feature, self.bundles_feature), 0)
        
        non_atom_users_feature, non_atom_bundles_feature,u_b = self.one_propagate(
            self.non_atom_graph, self.users_feature, self.bundles_feature, self.dnns_non_atom)
        
        # mx_users_feature=self.interpolation_data_augmentation(atom_users_feature,non_atom_users_feature)
        mx_users_feature=self.RepMixup(atom_users_feature,non_atom_users_feature)
        #mx_users_feature=atom_users_feature
        #mx_users_feature=non_atom_users_feature
        # mx_share,mx_specific=self.disentangle(mx_users_feature)
        # non_irr,non_spe=self.disentangle(non_atom_users_feature)
        # a_irr,a_spe=self.disentangle(atom_users_feature)


        # atom_user=torch.cat((a_irr,a_spe,mx_share),1)
        # non_atom_user=torch.cat((non_irr,non_spe,mx_share),1)
        # atom_user=self.fc(atom_user)
        # non_atom_user=self.fc(non_atom_user)
        # atom_user=F.normalize(self.mess_dropout(atom_user))
        # non_atom_user=F.normalize(self.mess_dropout(non_atom_user))


        # atom_att_user=self.adapter_(atom_user)
        # non_att_user=self.adapter_(non_atom_user)
        # non_att_user=F.normalize(self.mess_dropout(non_att_user))
        
        # dc_ir=torch.cat((a_irr,non_irr,mx_share),1)
        # dc_sp=torch.cat((a_spe,non_spe,mx_specific),1)
        # dc_ir=F.normalize(self.mess_dropout(dc_ir))
        # dc_sp=F.normalize(self.mess_dropout(dc_sp))

        # x = torch.cat((atom_users_feature,atom_items_feature),0)
        x = torch.cat((mx_users_feature,atom_items_feature),0)
        features_ui=self.adapter1(x)
        features_ui=F.normalize(self.mess_dropout(features_ui))
        A_feature, B_feature = torch.split(
             features_ui, (mx_users_feature.shape[0],atom_items_feature.shape[0]), 0)
        # atom_bundles_feature=self.pre_bundle(i)
        atom_bundles_feature = F.normalize(torch.matmul(self.pooling_graph,B_feature))
    
        # users_feature = [A_feature, mx_users_feature]

        # y = torch.cat((mx_users_feature,non_atom_bundles_feature),0)
        # features_ub=self.adapter2(y)
        # features_ub=F.normalize(self.mess_dropout(features_ub))
        # C_feature, D_feature = torch.split(
        #      features_ub, (mx_users_feature.shape[0],non_atom_bundles_feature.shape[0]), 0)
    
        # atom_bundles_feature=self.pre_bundle(i)
        # users_feature = [A_feature, non_atom_users_feature]
        users_feature = [A_feature, mx_users_feature]
        bundles_feature = [atom_bundles_feature, non_atom_bundles_feature]
        
        return users_feature, bundles_feature,u_i,u_b
   
#    def create_cor_loss(self, cor_u_embeddings):
#         cor_loss = torch.zeros(1).to(self.device)
#         cor_u_embeddings=cor_u_embeddings[:, 0, :]
#         ui_embeddings = cor_u_embeddings
#         ui_factor_embeddings = torch.split(ui_embeddings, int(ui_embeddings.shape[1] / self.n_factors), 1)

#         for i in range(0, self.n_factors - 1):
#             x = ui_factor_embeddings[i]
#             y = ui_factor_embeddings[i + 1]
#             cor_loss += self._create_distance_correlation(x, y)

#         cor_loss /= ((self.n_factors + 1.0) * self.n_factors / 2)

#         return cor_loss
   def graphEmbeddingNetwork(self,input_dim, embedding_dim,graph,device):
       linear = nn.Linear(input_dim, embedding_dim).to(device)
       emb = linear(graph).to(device)
       return emb

   def graphContrastiveLoss(self,graph_emb1,graph_emb2):
       graph_emb1=self.graphEmbeddingNetwork(graph_emb1.size(1),graph_emb1.size(1),graph_emb1,self.device)
       graph_emb2=self.graphEmbeddingNetwork(graph_emb2.size(1),graph_emb2.size(1),graph_emb2,self.device)
       euclidean_distance = F.pairwise_distance(graph_emb1, graph_emb1)
       loss_contrastive = torch.mean(torch.pow(euclidean_distance, 2))
       return loss_contrastive

   
   
   def cal_c_loss(self, pos, aug):
        # pos: [batch_size, :, emb_size]
        # aug: [batch_size, :, emb_size]
        pos = pos[:, 0, :]
        aug = aug[:, 0, :]
        
        pos = F.normalize(pos, p=2, dim=1)
        aug = F.normalize(aug, p=2, dim=1)
       
        pos_score = torch.sum(pos * aug, dim=1) # [batch_size]
        ttl_score = torch.matmul(pos, aug.permute(1, 0)) # [batch_size, batch_size]

        pos_score = torch.exp(pos_score / self.c_temp) # [batch_size]
        ttl_score = torch.sum(torch.exp(ttl_score / self.c_temp), axis=1) # [batch_size]

        c_loss = - torch.mean(torch.log(pos_score / ttl_score))

        return c_loss
   
   def create_intent_loss(self, u_emb_global, u_emb_loc):
        ui_gfactor_embeddings = torch.stack(torch.split(u_emb_global, int(u_emb_global.shape[1] / self.n_factors), 1),
                                            dim=1)
        #ui_gfactor_embeddings = F.normalize(ui_gfactor_embeddings, dim=2)
        ui_lfactor_embeddings = torch.stack(torch.split(u_emb_loc, int(u_emb_loc.shape[1] / self.n_factors), 1), dim=1)
        #ui_lfactor_embeddings = F.normalize(ui_lfactor_embeddings, dim=2)
        
        intent_loss = torch.log(torch.exp((ui_gfactor_embeddings * ui_lfactor_embeddings).sum(dim=2)) / torch.exp(
            torch.matmul(ui_gfactor_embeddings, torch.transpose(ui_lfactor_embeddings, -1, -2))).sum(dim=2)).mean(
            dim=1).mean()

        return intent_loss

   def cal_bpr_loss(self,pred):
        # pred: [bs, 1+neg_num]
        negs = pred[:, 1].unsqueeze(1)
        pos = pred[:, 0].unsqueeze(1)

        loss = - torch.log(torch.sigmoid(pos - negs)) # [bs]
        loss = torch.mean(loss)

        return loss

   def cal_loss(self, users_feature, bundles_feature):
        users_feature_atom, users_feature_non_atom = users_feature # batch_n_f
        bundles_feature_atom, bundles_feature_non_atom = bundles_feature # batch_n_f
        pred = torch.sum(users_feature_atom * bundles_feature_atom, 2) \
            + torch.sum(users_feature_non_atom * bundles_feature_non_atom, 2)
        
        bpr_loss = self.cal_bpr_loss(pred)

        
        
        # cl is abbr. of "contrastive loss"
        u_cross_view_cl = self.cal_c_loss(users_feature_atom, users_feature_non_atom)
        b_cross_view_cl = self.cal_c_loss(bundles_feature_atom, bundles_feature_non_atom)

        u_create_intent = self.create_intent_loss(users_feature_atom, users_feature_non_atom)
        b_create_intent = self.create_intent_loss(bundles_feature_atom, bundles_feature_non_atom)
        
        
        
        c_losses = [u_cross_view_cl, b_cross_view_cl]
        

        c_loss = sum(c_losses) / len(c_losses)
        
        ub_losses=[u_create_intent,b_create_intent]
        ub_loss=sum(ub_losses) / len(ub_losses)
        

      

        return pred,bpr_loss, c_loss,ub_loss
   


   def forward(self, users, bundles):
      
        users_feature, bundles_feature,u_i,u_b = self.propagate()

        users_embedding = [i[users].expand(-1, bundles.shape[1], -1) for i in users_feature]
        bundles_embedding = [i[bundles] for i in bundles_feature]

        
        pre,bpr_loss, c_loss,ub_loss = self.cal_loss(users_embedding, bundles_embedding)
        graph_loss=self.graphContrastiveLoss(u_i,u_b)

        return pre,bpr_loss, c_loss,graph_loss,ub_loss


   def evaluate(self, propagate_result, users):
        users_feature, bundles_feature = propagate_result
        users_feature_atom, users_feature_non_atom = [i[users] for i in users_feature]
        bundles_feature_atom, bundles_feature_non_atom = bundles_feature

        scores = torch.mm(users_feature_atom, bundles_feature_atom.t()) + torch.mm(users_feature_non_atom, bundles_feature_non_atom.t())
        return scores


