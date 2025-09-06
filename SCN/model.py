import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F


######################## Layer #############################

class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()

        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, G: torch.Tensor):
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        x = G.matmul(x)
        return x

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

#         self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.relu = nn.LeakyReLU()

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.relu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime   
        
    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0] # number of nodes

        # Below, two matrices are created that contain embeddings in their rows in different orders.
        # (e stands for embedding)
        # These are the rows of the first matrix (Wh_repeated_in_chunks): 
        # e1, e1, ..., e1,            e2, e2, ..., e2,            ..., eN, eN, ..., eN
        # '-------------' -> N times  '-------------' -> N times       '-------------' -> N times
        # 
        # These are the rows of the second matrix (Wh_repeated_alternating): 
        # e1, e2, ..., eN, e1, e2, ..., eN, ..., e1, e2, ..., eN 
        # '----------------------------------------------------' -> N times
        # 
        
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        # Wh_repeated_in_chunks.shape == Wh_repeated_alternating.shape == (N * N, out_features)

        # The all_combination_matrix, created below, will look like this (|| denotes concatenation):
        # e1 || e1
        # e1 || e2
        # e1 || e3
        # ...
        # e1 || eN
        # e2 || e1
        # e2 || e2
        # e2 || e3
        # ...
        # e2 || eN
        # ...
        # eN || e1
        # eN || e2
        # eN || e3
        # ...
        # eN || eN

        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        # all_combinations_matrix.shape == (N * N, 2 * out_features)

        return all_combinations_matrix.view(N, N, 2 * self.out_features)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
    
class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)

    
class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)
                
        self.a = nn.Parameter(torch.zeros(size=(1, 2*out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.special_spmm = SpecialSpmm()

    def forward(self, input, adj):
        dv = 'cuda' if input.is_cuda else 'cpu'

        N = input.size()[0]
        edge = adj.nonzero().t()

        h = torch.mm(input, self.W)
        # h: N x out
        assert not torch.isnan(h).any()

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # edge: 2*D x E

        edge_e = torch.exp(-self.relu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N,1), device=dv))
        # e_rowsum: N x 1

        edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out
        
        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'    
    

######################## Model #############################

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = HGNN_conv(nfeat, nhid, 1)
        self.gc2 = HGNN_conv(nhid, nclass, 1)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1), x

class HGNN(nn.Module):
    def __init__(self, in_ch, n_class, n_hid, dropout):
        super(HGNN, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv(in_ch, n_hid)
        self.hgc2 = HGNN_conv(n_hid, n_class)

    def forward(self, x, G):
        x = F.relu(self.hgc1(x, G))
        x = F.dropout(x, self.dropout)
        x = self.hgc2(x, G)
        return x

class HCoN(nn.Module):
    def __init__(self, input_feat_x_dim, input_feat_y_dim, dim_l1, nclass, dropout):
        super(HCoN, self).__init__()
        
        self.dropout = dropout

        self.gcx1_part1 = HGNN_conv(input_feat_x_dim, dim_l1)
        self.gcx1_part2 = HGNN_conv(input_feat_y_dim, dim_l1)
        self.gcx2_part1 = HGNN_conv(dim_l1, nclass)
        self.gcx2_part2 = HGNN_conv(dim_l1, nclass)
        
        self.gcy1_part1 = HGNN_conv(input_feat_y_dim, dim_l1)
        self.gcy1_part2 = HGNN_conv(input_feat_x_dim, dim_l1)
        self.gcy2_part1 = HGNN_conv(dim_l1, nclass)
        self.gcy2_part2 = HGNN_conv(dim_l1, nclass)

    
    def forward(self, hx1, hx2, x0, hy1, hy2, y0, alpha, beta):       
        
        
        # layer 1
        x_part1 = self.gcx1_part1(x0, hx1) * alpha
        x_part2 = self.gcx1_part2(y0, hx2) * (1 - alpha)
        x1 = x_part1 + x_part2   
        x1 = F.relu(x1)
        x1 = F.dropout(x1, self.dropout)

        
        y_part1 = self.gcy1_part1(y0, hy1) * beta
        y_part2 = self.gcy1_part2(x0, hy2) * (1 - beta)
        y1 = y_part1 + y_part2
        y1 = F.relu(y1)  
        y1 = F.dropout(y1, self.dropout)
        
        
        # layer 2        
        x_part1 = self.gcx2_part1(x1, hx1) * alpha
        x_part2 = self.gcx2_part2(y1, hx2) * (1 - alpha)
        x2 = x_part1 + x_part2 
#         x2 = F.relu(x2)  

        y_part1 = self.gcy2_part1(y1, hy1) * alpha
        y_part2 = self.gcy2_part2(x1, hy2) * (1 - alpha)
        y2 = y_part1 + y_part2
        
        
        return x2, y2

class MLP(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(MLP, self).__init__()

        self.layer1 = nn.Linear(nfeat, nhid)
        self.layer2 = nn.Linear(nhid, nclass)
        self.dropout = dropout

    def forward(self, x):
        
        # layer1
        x = self.layer1(x)
        x = F.relu(x)
        
        # layer2
        x = self.layer2(x)

        return x   
    
    
class ours(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(ours, self).__init__()

        self.gc1 = HGNN_conv(nfeat, nhid, 1)
        self.gc2 = HGNN_conv(nhid, nclass, 1)
        self.dropout = dropout

    def forward(self, x1, adj1, x2, adj2, alpha, indices):
        
        # layer1
        x1 = self.gc1(x1, adj1)        
        x2 = self.gc1(x2, adj2)

        x = alpha * x1 + (1 - alpha) * x2
        x = F.relu(x)
        x = F.dropout(x, self.dropout)
            
            
        # 得到新的X Y Z --> S1 = x; S2 = [Z; Y; X]，再放入第二层
        x1 = self.gc2(x, adj1)
        x_t = torch.index_select(x, 0, indices)
        x2 = self.gc2(x_t, adj2)

        x = alpha * x1 + (1 - alpha) * x2
#         x = F.relu(x)

        return x   


class ours_2layer_dropout(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(ours_2layer_dropout, self).__init__()

        self.gc1 = HGNN_conv(nfeat, nhid, 1)
        self.gc2 = HGNN_conv(nhid, nclass, 1)
        self.dropout = dropout

    def forward(self, x1, adj1, x2, adj2, alpha, indices):
        
        # layer1
        x1 = self.gc1(x1, adj1)        
        x2 = self.gc1(x2, adj2)

        x = alpha * x1 + (1 - alpha) * x2
        x = F.relu(x)
        x = F.dropout(x, self.dropout)
            
            
        # 得到新的X Y Z --> S1 = x; S2 = [Z; Y; X]，再放入第二层
        x1 = self.gc2(x, adj1)
        x_t = torch.index_select(x, 0, indices)
        x2 = self.gc2(x_t, adj2)

        x = alpha * x1 + (1 - alpha) * x2
        x = F.dropout(x, self.dropout)
#         x = F.relu(x)

        return x


class ours_extend(nn.Module):
    # alpha如果是向量，即每个simplex聚合都有各自权重
    def __init__(self, nfeat0, nfeat1, nfeat2, nhid, nclass, dropout):
        super(ours_extend, self).__init__()
        
        self.gc0 = HGNN_conv(nfeat0, nclass, 1)
        self.gc1 = HGNN_conv(nfeat1, nclass, 1)
        self.gc2 = HGNN_conv(nfeat2, nclass, 1)
        
        self.gc = HGNN_conv(nhid, nclass, 1)
        self.dropout = dropout

    def forward(self, x, y, z, H01, H02, h01, h02, indices, alpha, beta):
        
        # layer1
        x1 = self.gc0(x, H01)        
        x2 = self.gc1(y, h01)
        x3 = self.gc0(x, H02)
        x4 = self.gc2(z, h02)

        x = alpha * x1 + (1 - alpha) * x2 + beta * x3 + (1 - beta) * x4
        x = F.relu(x)
#         x = F.dropout(x, self.dropout)
            
            
#         # 第二层
#         x1 = self.gc(x, H01)        
#         x2 = self.gc(y, h01)
#         x3 = self.gc(x, H02)
#         x4 = self.gc(z, h02)

#         x = x1 + x2
# #         x = F.relu(x)

        return x     
    
    
class ours2(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(ours2, self).__init__()

        self.gc1 = HGNN_conv(nfeat, nhid, 1)
        self.gc2 = HGNN_conv(nhid, nclass, 1)
        self.dropout = dropout

    def forward(self, x1, adj1, x2, adj2, alpha, indices):
        
        # layer1
        x1 = self.gc1(x1, adj1)        
        x2 = self.gc1(x2, adj2)

        x = alpha * x1 + (1 - alpha) * x2
        x = F.relu(x)
        x = F.dropout(x, self.dropout)
            
            
        # 得到新的X Y Z --> S1 = x; S2 = [Z; Y; X]，再放入第二层
        x1 = self.gc2(x, adj1)
        x_t = torch.index_select(x, 0, indices)
        x2 = self.gc2(x_t, adj2)

        x = alpha * x1 + (1 - alpha) * x2
        x = F.relu(x)

        return x       

class HCNH_TKDD(nn.Module):
    def __init__(self, input_feat_x_dim, input_feat_y_dim, dim_l1, nclass, dropout):
        super(HCNH_TKDD, self).__init__()

        self.dropout = dropout

        
        self.gcx1 = HGNN_conv(input_feat_x_dim, dim_l1)
        self.gcx2 = HGNN_conv(dim_l1, nclass)

        self.gcy1 = HGNN_conv(input_feat_y_dim, dim_l1)
        self.gcy2 = HGNN_conv(dim_l1, nclass)
            

    def forward(self, x, hx, y, hy):
        
        ######################## filtering on x ########################
        x = self.gcx1(x, hx)
        x = F.relu(x)
        x = F.dropout(x, self.dropout)
 
        x = self.gcx2(x, hx)
        x = F.relu(x)
        
        
        ######################## filtering on y ########################
        y = self.gcy1(y, hy)
        y = F.relu(y)  
        y = F.dropout(y, self.dropout)

        y = self.gcy2(y, hy)
        y = F.relu(y)

        
        
        ######################## decode h ########################
        h = torch.mm(x, y.t())
        h = torch.sigmoid(h)
        
        
        ######################## cross-entropy  ########################
#         output = F.log_softmax(x, dim=1)
        
        return h, x   
    
class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1), x    
    
    
class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nheads):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout

        self.attentions = [SpGraphAttentionLayer(nfeat, nhid, dropout=dropout, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1), x    
    
    
    
    
    
    
    
    
    
    