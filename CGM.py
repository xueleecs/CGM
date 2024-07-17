import numpy as np
from torch.utils.data import DataLoader
import torch
from torch import nn
from torch.autograd import Variable
from itertools import chain
from sklearn import metrics
import torch.nn.functional as F
from sklearn import preprocessing
import warnings
from Data_loader import *
import time

warnings.filterwarnings("ignore")
torch.backends.cudnn.enabled = False
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from transformers import BertModel, BertTokenizer
import re
from Bert import *


def Indd_train(clf, train_data, test_data, OneEncoded):
    train_data_ = [(np.array([np.concatenate((OneEncoded[train_data[i][0]], train_data[i][3][0]), axis=0),
                              np.concatenate((OneEncoded[train_data[i][1]], train_data[i][3][1]), axis=0)]),
                    train_data[i][2]) for i in range(len(train_data))]
    test_data_ = [(np.array([np.concatenate((OneEncoded[test_data[i][0]], test_data[i][3][0]), axis=0),
                             np.concatenate((OneEncoded[test_data[i][1]], test_data[i][3][1]), axis=0)]),
                   test_data[i][2]) for i in range(len(test_data))]


    net = clf(Nodecount, args.nhid0, args.nhid1, args.dropout, args.alpha)
    criterion = nn.CrossEntropyLoss() 
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    print("===========================================")
    train_data_ = [(torch.from_numpy(train_data_[i][0].astype(float)), torch.from_numpy(np.array(train_data_[i][1]))) for i in
                   range(len(train_data_))]
    test_data_ = [(torch.from_numpy(test_data_[i][0].astype(float)), torch.from_numpy(np.array(test_data_[i][1]))) for i in
                  range(len(test_data_))]

    save_figdata = [[i, test_data_[i][1]] for i in range(len(test_data_))]
    train_data_ = DataLoader(train_data_, batch_size=256, shuffle=True)
    test_data_ = DataLoader(test_data_, batch_size=128, shuffle=False)

    _, epochspred = train(net, train_data_, test_data_, num_epochs, optimizer, criterion, scheduler)

    save_figdata = np.mat(save_figdata)
    save_figdata = np.hstack((save_figdata, epochspred))
    np.savetxt("/result/" + disease + "+net.txt", save_figdata, fmt='%0.6f')



def train(net, train_data, valid_data, num_epochs, optimizer, criterion, scheduler):

    if torch.cuda.is_available():
        net = net.cuda()
    output_valid_data = []
    epochspred = []
    epochspredlabel = []
    predic_list = []
    lable_list=[]    
    ######################
    Mloss_reg=1000
    for epoch in range(num_epochs):
        start_onebs = time.time()
        train_loss = 0
        train_acc = 0
        net = net.train()

        for a, a_label in train_data:
            a_label = a_label.long()
            a = torch.tensor(a, dtype=torch.float32)
            if torch.cuda.is_available():
                a = Variable(a.cuda())  # (64, 2, 128, 128)   (256,2,366)
                a_label = Variable(a_label.cuda())  # (64, 128, 128)  (256,none)
            else:
                a = Variable(a)  # Ideally, the dimension of a is 2*128*128
                a_label = Variable(a_label)
            output, output_label, L_1st, L_2nd, L_all,predic,LCON = net(a)

            L_reg = 0
            for param in net.parameters():
                L_reg += args.nu1 * torch.sum(torch.abs(param)) + args.nu2 * torch.sum(param * param)
            loss = criterion(output, a_label) + L_all + L_reg#+LCON
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step(epoch)
        if valid_data is not None:
            Acc = 0
            Recall = 0
            Specificity = 0
            Precision = 0
            MCC = 0
            F1 = 0
            AUC = 0
            net = net.eval()
            loss_sum, loss_L1, loss_L2, loss_reg = 0, 0, 0, 0
            onepred = []
            
            for a, a_label in valid_data:
                a_label = a_label.long()
                a = torch.tensor(a, dtype=torch.float32)
                if torch.cuda.is_available():
                    a = Variable(a.cuda(), volatile=True)  # (64, 2, 128, 128)
                    a_label = Variable(a_label.cuda(), volatile=True)  # (64, 128, 128)
                else:
                    a = Variable(a, volatile=True)
                    a_label = Variable(a_label, volatile=True)
                output, output_label, L_1st, L_2nd, L_all,predic,LCON = net(a)

                if epoch in range (450,500):

                    predic_list.extend(predic)
                    lable_list.extend(a_label)

                pred_label_valid_data = output.max(1)[1]
                if epoch == num_epochs - 1:
                    output_valid_data.append(pred_label_valid_data)
                L_reg = 0
                for param in net.parameters():
                    L_reg += args.nu1 * torch.sum(torch.abs(param)) + args.nu2 * torch.sum(param * param)
                loss = criterion(output, a_label) + L_all + L_reg#+LCON
                loss.item()
                y_pred = output.max(1)[1]

                onepred.append(output[:, 1].tolist())
                onepred = getnewList(onepred)

                Acc += metrics.accuracy_score(a_label.cpu(), y_pred.cpu())
                Recall += metrics.recall_score(a_label.cpu(), y_pred.cpu())
                Precision += metrics.precision_score(a_label.cpu(), y_pred.cpu())
                F1 += metrics.f1_score(a_label.cpu(), y_pred.cpu())
                AUC += metrics.roc_auc_score(a_label.cpu(), y_pred.cpu())

                loss_sum += loss
                loss_L1 += L_1st
                loss_L2 += L_2nd
                loss_reg += L_reg

                if loss< Mloss_reg:
                    Mloss_reg=loss

            epoch_str = (
                    "Epoch %d. Accuracy: %f, Recall: %f, Precision: %f, F1: %f, AUC: %f"
                    % (epoch, Acc / len(valid_data), Recall / len(valid_data),
                       Precision / len(valid_data), F1 / len(valid_data),
                       AUC / len(valid_data)))
            end_onebs = time.time()
            epoch_time = int(end_onebs - start_onebs)
            epoch_loss = ('{0}s - loss: {1: .4f} - 2nd_loss: {2: .4f} - 1st_loss: {3: .4f} - loss_reg: {4: .4f}'
                          .format(epoch_time, loss_sum / len(valid_data), loss_L2 / len(valid_data),
                                  loss_L1 / len(valid_data), loss_reg / len(valid_data)))

        else:
            epoch_str = ("Epoch %d. Train Loss: %f, Train Acc: %f, " %
                         (epoch, train_loss / len(train_data),
                          train_acc / len(train_data)))
        print(epoch_str)
        print(epoch_loss)
        epochspred.append(onepred)


    output_valid_data = [output_valid_data[i] for i in range(len(output_valid_data))]
    output_valid_data = np.array(list(chain(*[x.cpu().numpy() for x in output_valid_data])))
    epochspred = np.mat(epochspred).T


    print("Diabetes" +"all ")
    print("Loss" + str(Mloss_reg))

    return output_valid_data, epochspred


# Models
def squash(inputs, axis=-1):
    """
    The non-linear activation used in Capsule. It drives the length of a large vector to near 1 and small vector to 0
    :param inputs: vectors to be squashed
    :param axis: the axis to squash
    :return: a Tensor with same size as inputs
    """
    norm = torch.norm(inputs, p=2, dim=axis, keepdim=True)
    scale = norm**2 / (1 + norm**2) / (norm + 1e-8)
    return scale * inputs


class DenseCapsule(nn.Module):
    """
    The dense capsule layer. It is similar to Dense (FC) layer. Dense layer has `in_num` inputs, each is a scalar, the
    output of the neuron from the former layer, and it has `out_num` output neurons. DenseCapsule just expands the
    output of the neuron from scalar to vector. So its input size = [None, in_num_caps, in_dim_caps] and output size = \
    [None, out_num_caps, out_dim_caps]. For Dense Layer, in_dim_caps = out_dim_caps = 1.

    :param in_num_caps: number of cpasules inputted to this layer
    :param in_dim_caps: dimension of input capsules
    :param out_num_caps: number of capsules outputted from this layer
    :param out_dim_caps: dimension of output capsules
    :param routings: number of iterations for the routing algorithm
    """
    def __init__(self, in_num_caps, in_dim_caps, out_num_caps, out_dim_caps, routings=3):
        super(DenseCapsule, self).__init__()
        self.in_num_caps = in_num_caps
        self.in_dim_caps = in_dim_caps
        self.out_num_caps = out_num_caps
        self.out_dim_caps = out_dim_caps
        self.routings = routings
        self.weight = nn.Parameter(0.01 * torch.randn(out_num_caps, in_num_caps, out_dim_caps, in_dim_caps))

    def forward(self, x):
        # print(x.shape) #[32, 32, 8]
        # print(x[:, None, :, :, None].shape) #[32, 1, 32, 8, 1]
        # print(self.weight.shape) #[203, 1152, 16, 8]
        # x.size=[batch, in_num_caps, in_dim_caps]
        # expanded to    [batch, 1,            in_num_caps, in_dim_caps,  1]
        # weight.size   =[       out_num_caps, in_num_caps, out_dim_caps, in_dim_caps]
        # torch.matmul: [out_dim_caps, in_dim_caps] x [in_dim_caps, 1] -> [out_dim_caps, 1]
        # => x_hat.size =[batch, out_num_caps, in_num_caps, out_dim_caps]
        x_hat = torch.squeeze(torch.matmul(self.weight, x[:, None, :, :, None]), dim=-1)

        # In forward pass, `x_hat_detached` = `x_hat`;
        # In backward, no gradient can flow from `x_hat_detached` back to `x_hat`.
        x_hat_detached = x_hat.detach()

        # The prior for coupling coefficient, initialized as zeros.
        # b.size = [batch, out_num_caps, in_num_caps]
        b = Variable(torch.zeros(x.size(0), self.out_num_caps, self.in_num_caps))

        assert self.routings > 0, 'The \'routings\' should be > 0.'
        for i in range(self.routings):
            # c.size = [batch, out_num_caps, in_num_caps]
            c = F.softmax(b, dim=1)

            # At last iteration, use `x_hat` to compute `outputs` in order to backpropagate gradient
            if i == self.routings - 1:
                # c.size expanded to [batch, out_num_caps, in_num_caps, 1           ]
                # x_hat.size     =   [batch, out_num_caps, in_num_caps, out_dim_caps]
                # => outputs.size=   [batch, out_num_caps, 1,           out_dim_caps]
                outputs = squash(torch.sum(c[:, :, :, None] * x_hat, dim=-2, keepdim=True))
                # outputs = squash(torch.matmul(c[:, :, None, :], x_hat))  # alternative way
            else:  # Otherwise, use `x_hat_detached` to update `b`. No gradients flow on this path.
                outputs = squash(torch.sum(c[:, :, :, None] * x_hat_detached, dim=-2, keepdim=True))
                # outputs = squash(torch.matmul(c[:, :, None, :], x_hat_detached))  # alternative way

                # outputs.size       =[batch, out_num_caps, 1,           out_dim_caps]
                # x_hat_detached.size=[batch, out_num_caps, in_num_caps, out_dim_caps]
                # => b.size          =[batch, out_num_caps, in_num_caps]
                b = b + torch.sum(outputs * x_hat_detached, dim=-1)

        return torch.squeeze(outputs, dim=-2)


class PrimaryCapsule(nn.Module):
    """
    Apply Conv2D with `out_channels` and then reshape to get capsules
    :param in_channels: input channels
    :param out_channels: output channels
    :param dim_caps: dimension of capsule
    :param kernel_size: kernel size
    :return: output tensor, size=[batch, num_caps, dim_caps]
    """
    def __init__(self, in_channels, out_channels, dim_caps, kernel_size, stride=1, padding=0):
        super(PrimaryCapsule, self).__init__()
        self.dim_caps = dim_caps
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        # print(x.shape) #[32, 256, 37, 1]
        outputs = self.conv2d(x)
        outputs = outputs.view(x.size(0), -1, self.dim_caps)#[256,32,8]
        return squash(outputs)

class CapsuleNet(nn.Module):
    """
    A Capsule Network on MNIST.
    :param input_size: data size = [channels, width, height]
    :param classes: number of classes
    :param routings: number of routing iterations
    Shape:
        - Input: (batch, channels, width, height), optional (batch, classes) .
        - Output:((batch, classes), (batch, channels, width, height))
    """
    def __init__(self, input_size, classes, routings):
        super(CapsuleNet, self).__init__()
        self.input_size = input_size
        self.classes = classes
        self.routings = routings

        # Layer 1: Just a conventional Conv2D layer
        self.conv1 = nn.Conv2d(input_size[0], 256, kernel_size=(32, 32), stride=1, padding=0)

        # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_caps, dim_caps]
        self.primarycaps = PrimaryCapsule(256, 256, 8, kernel_size=(1, 1), stride=2, padding=0)

        # Layer 3: Capsule layer. Routing algorithm works here.
        self.digitcaps = DenseCapsule(in_num_caps=32, in_dim_caps=8,
                                      out_num_caps=classes, out_dim_caps=16, routings=routings)

        # Decoder network.
        self.decoder = nn.Sequential(
            nn.Linear(16*classes, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, input_size[0] * input_size[1] * input_size[2]),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU()

    def forward(self, x, y=None): 
        x = self.relu(self.conv1(x)) 
        x = self.primarycaps(x)
        x = self.digitcaps(x)
        length = x.norm(dim=-1)
        if y is None:  
            index = length.max(dim=1)[1]
            y = Variable(torch.zeros(length.size()).scatter_(1, index.view(-1, 1).cpu().data, 1.))#[16,203]
        reconstruction = self.decoder((x * y[:, :, None]).view(x.size(0), -1))#[16,8000] [256,1024]
        x_recon=reconstruction.view(-1, *self.input_size)
        L_recon=nn.MSELoss()(x_recon, x1)

        return length, x_recon, L_recon
    
class layer_normalization(nn.Module):
    def __init__(self, features, epsilon=1e-8):
        super(layer_normalization, self).__init__()
        self.epsilon = epsilon
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.epsilon) + self.beta


class multihead_attention(nn.Module):
    def __init__(self, num_units, num_heads=4, dropout_rate=0, causality=False):
        super(multihead_attention, self).__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.causality = causality
        self.Q_proj = nn.Sequential(nn.Linear(self.num_units, self.num_units), nn.ReLU())
        self.K_proj = nn.Sequential(nn.Linear(self.num_units, self.num_units), nn.ReLU())
        self.V_proj = nn.Sequential(nn.Linear(self.num_units, self.num_units), nn.ReLU())
        self.output_dropout = nn.Dropout(p=self.dropout_rate)
        self.normalization = layer_normalization(self.num_units)

    def forward(self, queries, keys, values):
        # keys, values: same shape of [N, T_k, C_k]
        # queries: A 3d Variable with shape of [N, T_q, C_q]
        # Linear projections
        if torch.cuda.is_available():
            queries = queries.cuda()
            keys = keys.cuda()
            values = values.cuda()
        Q = self.Q_proj(queries)  # (N, T_q, C)     
        K = self.K_proj(keys)  # (N, T_q, C)       
        V = self.V_proj(values)  # (N, T_q, C)      
        # Split and concat
        Q_ = torch.cat(torch.chunk(Q, self.num_heads, dim=2), dim=0)  # (h*N, T_q, C/h)     
        K_ = torch.cat(torch.chunk(K, self.num_heads, dim=2), dim=0)  # (h*N, T_q, C/h)     
        V_ = torch.cat(torch.chunk(V, self.num_heads, dim=2), dim=0)  # (h*N, T_q, C/h)     
        # Multiplication
        outputs = torch.bmm(Q_, K_.permute(0, 2, 1))  # (h*N, T_q, T_k)
        # Scale
        outputs = outputs / (K_.size()[-1] ** 0.5)
        # Key Masking
        key_masks = torch.sign(torch.abs(torch.sum(keys, dim=-1)))  # (N, T_k)
        key_masks = key_masks.repeat(self.num_heads, 1)  # (h*N, T_k)
        key_masks = torch.unsqueeze(key_masks, 1).repeat(1, queries.size()[1], 1)  # (h*N, T_q, T_k)
        if torch.cuda.is_available():
            padding = Variable(torch.ones(*outputs.size()) * (-2 ** 32 + 1)).cuda()
        else:
            padding = Variable(torch.ones(*outputs.size()) * (-2 ** 32 + 1))
        # print(padding.device)
        condition = key_masks.eq(0.).float()
        # print(condition.device)
        outputs = padding * condition + outputs * (1. - condition)
        # Causality = Future blinding
        if self.causality:
            diag_vals = torch.ones(*outputs[0, :, :].size())  # (T_q, T_k)
            tril = torch.tril(diag_vals, diagonal=0)  # (T_q, T_k)
            # print(tril)
            masks = Variable(torch.unsqueeze(tril, 0).repeat(outputs.size()[0], 1, 1))  # (h*N, T_q, T_k)
            padding = Variable(torch.ones(*masks.size()) * (-2 ** 32 + 1))
            condition = masks.eq(0.).float()
            outputs = padding * condition + outputs * (1. - condition)
        # Activation
        outputs = F.softmax(outputs, dim=-1)  # (h*N, T_q, T_k)
        # Query Masking
        query_masks = torch.sign(torch.abs(torch.sum(queries, dim=-1)))  # (N, T_q)
        query_masks = query_masks.repeat(self.num_heads, 1)  # (h*N, T_q)
        query_masks = torch.unsqueeze(query_masks, 2).repeat(1, 1, keys.size()[1])  # (h*N, T_q, T_k)
        outputs = outputs * query_masks
        # Dropouts
        outputs = self.output_dropout(outputs)  # (h*N, T_q, T_k)
        # Weighted sum
        outputs = torch.bmm(outputs, V_)  # (h*N, T_q, C/h)
        # Restore shape
        outputs = torch.cat(torch.chunk(outputs, self.num_heads, dim=0), dim=2)  # (N, T_q, C)
        # Residual connection
        outputs += queries
        # Normalize
        outputs = self.normalization(outputs)  # (N, T_q, C)
        return outputs


def block(in_channel, out_channel, stride=1):
    layer = nn.Sequential(nn.BatchNorm2d(in_channel),
                          nn.Conv2d(in_channel, out_channel, 3, stride=stride, padding=1, bias=False))
    return layer


class cnn_block(nn.Module):
    def __init__(self, in_channel, out_channel, same_shape=True):
        super(cnn_block, self).__init__()
        self.same_shape = same_shape
        stride = 1 if self.same_shape else 2
        self.conv1 = block(in_channel, out_channel, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = block(out_channel, out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        # if not self.same_shape:
        self.conv3 = nn.Conv2d(in_channel, out_channel, 1, stride=stride)

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        x = self.conv1(x)
        x = F.relu(self.bn1(x), True)
        x = self.conv2(x)
        x = F.relu(self.bn2(x), True)
        x = self.conv3(x)
        return x


class cnn2_block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(cnn2_block, self).__init__()
        self.conv1 = block(in_channel, out_channel)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = block(out_channel, out_channel)

    def forward(self, x):
        x1 = torch.tensor(x, dtype=torch.float32)
        x = self.conv1(x1)
        x = F.relu(self.bn1(x), True)
        x = self.conv2(x)
        # x = x*x
        # x = x1+x
        # x = F.relu(self.bn1(x),True)
        return x

class cnn3_block(nn.Module):
    def __init__(self,in_channel, out_channel):
        super(cnn3_block, self).__init__()
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv = nn.Conv2d(1, 1, kernel_size=(8, 22), stride=(8, 22), padding=0, bias=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = F.relu(self.bn1(x), True)
        return x
    
class ModelCon(nn.Module):
    def __init__(self, node_size, nhid0, nhid1, droput, alpha):
        super(ModelCon, self).__init__()
        self.attention = multihead_attention(32)
        self.pro_len = 32
        self.cos_sim = nn.CosineSimilarity()
        self.Bilinear_sim = nn.Bilinear(2 * 32, 2 * 32, 1, bias=False)
        self.linear_sim = nn.Linear(4 * 32, 1)
        self.dense_dim1 = 512
        self.dense_dim2 = 256
        self.dense_dim3 = 128

        self.encode0 = nn.Linear(node_size * 2, nhid0)
        self.encode1 = nn.Linear(nhid0, nhid1)
        self.decode0 = nn.Linear(nhid1, nhid0)
        self.decode1 = nn.Linear(nhid0, node_size * 2)
        self.droput = droput
        self.alpha = alpha

        self.c = Nodecount
        self.clf_layer = nn.Sequential(nn.Linear(179456 * 2 + 4 * 32 + 3+ self.c*2, self.dense_dim1),  
                                       nn.BatchNorm1d(self.dense_dim1), nn.Dropout(0.2),  # nn.ELU(),
                                       nn.Linear(self.dense_dim1, self.dense_dim2), nn.ELU(),
                                       nn.Linear(self.dense_dim2, self.dense_dim3),
                                       nn.Linear(self.dense_dim3, 2), nn.BatchNorm1d(2))


    def CapsuleNet(self,input_):
        input_ = torch.tensor(input_, dtype=torch.float32).cpu()
        netliearn1 = nn.Linear(179456, 1 * 32 * 32)
        input_0 = netliearn1(input_)
        input_0 = input_0.reshape(len(input_0), 1, 32, 32)
        nett = CapsuleNet([1,32,32],2,10)
        y_pred, x_recon,L_con = nett(input_0)

        return y_pred, x_recon,L_con

    def aggregate_layers(self, input_):
        "Apply layers to input then concatenate result"
        avglayer = nn.AvgPool1d(self.pro_len)
        maxlayer = nn.MaxPool1d(self.pro_len)
        avg_ = avglayer(input_.transpose(1, 2)).squeeze()
        max_ = maxlayer(input_.transpose(1, 2)).squeeze()
        out_ = torch.cat((avg_, max_), 1)
        return out_

    def forward(self, X):
        # print(X.split(1, 1))
        X1, X2 = X.split(1, 1)  # proteinEncoded1: torch.Size([512, 1, 1218])

        proteinEncoded1, adj1 = torch.split(X1, split_size_or_sections=[179456, self.c], dim=2)#LX-[173,1,179456]
        proteinEncoded2, adj2 = torch.split(X2, split_size_or_sections=[179456, self.c], dim=2)

        y11, proCaps_1,L_con1 = self.CapsuleNet(proteinEncoded1)#[256,1,32,32]
        proGatedCNN_1=proCaps_1.squeeze(1)#[256,32,32]
        y11, proCaps_2,L_con2 = self.CapsuleNet(proteinEncoded2)
        proGatedCNN_2=proCaps_2.squeeze(1)



        #######################################################################################
        conc_pro12 = torch.cat([proteinEncoded1, proteinEncoded2], dim=2).squeeze(1)
        sonsen = conc_pro12.cuda().data.cpu().numpy()

        sonsen = preprocessing.normalize(sonsen)
        sonsen = preprocessing.scale(sonsen)
        sonsen = torch.from_numpy(np.array(sonsen))
        sonsen = sonsen.float()

        # network
        adj_batch = torch.cat([adj1, adj2], axis=2).squeeze(1)  # [256,730]+[256,730]=[256, 1460]
        b_mat = torch.ones_like(adj_batch)  # [256, 1460]
        b_mat[adj_batch != 0] = args.beta

        t0 = F.leaky_relu(self.encode0(adj_batch))#[256,1000]
        t0 = F.leaky_relu(self.encode1(t0))#[256,128]
        embedding = t0  # [256, 128]
        t0 = F.leaky_relu(self.decode0(t0))#[256,1000]
        t0 = F.leaky_relu(self.decode1(t0))# [256, 1460]
        embedding_norm = torch.sum(embedding * embedding, dim=1, keepdim=True)  # [256, 1]

        L_1st = torch.sum((embedding_norm -
                           2 * torch.mm(embedding, torch.transpose(embedding, dim0=0, dim1=1))
                           + torch.transpose(embedding_norm, dim0=0, dim1=1)))
        L_2nd = torch.sum(((adj_batch - t0) * b_mat) * ((adj_batch - t0) * b_mat))

        # attention
        pro1_att = self.attention(proGatedCNN_1, proGatedCNN_2, proGatedCNN_1)  
        pro2_att = self.attention(proGatedCNN_2, proGatedCNN_1, proGatedCNN_2)
        pro1_self = torch.mul(proGatedCNN_1.cuda(), pro1_att.cuda())  
        pro2_self = torch.mul(proGatedCNN_2.cuda(), pro2_att.cuda())
        pro1_rep = self.aggregate_layers(pro1_self)  
        pro2_rep = self.aggregate_layers(pro2_self)
        sen_cos_sim = self.cos_sim(pro1_rep, pro2_rep).view(-1, 1).float()  
        sen_bilinear_sim = self.Bilinear_sim(pro1_rep, pro2_rep)  
        sen_linear_sim = torch.tanh(self.linear_sim(torch.cat((pro1_rep, pro2_rep), -1)))  
        merged = torch.cat((
            pro1_rep.cuda(), pro2_rep.cuda(), sonsen.cuda(),sen_cos_sim.cuda(), sen_bilinear_sim.cuda(),
            sen_linear_sim.cuda(), t0.cuda()), 1)  
        merged = merged.cuda().data.cpu().numpy()
        merged = preprocessing.normalize(merged)
        merged = preprocessing.scale(merged)
        merged1 = preprocessing.scale(merged)
        merged = torch.from_numpy(np.array(merged))
        merged = merged.float()
        output = self.clf_layer(merged.cuda())  
        pred = torch.sigmoid(output).squeeze()
        _, m_label = pred.max(1)
        LCON=(L_con2+L_con2)*0.0005
        return pred, m_label, L_1st, self.alpha * L_2nd, L_1st + self.alpha * L_2nd,merged1,LCON


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument('--workers', default=8, type=int,
                        help='Number of parallel processes.')
    parser.add_argument('--weighted', action='store_true', default=False,
                        help='Treat graph as weighted')
    parser.add_argument('--epochs', default=100, type=int,
                        help='The training epochs of SDNE')
    parser.add_argument('--dropout', default=0.5, type=float,
                        help='Dropout rate (1 - keep probability)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='Weight for L2 loss on embedding matrix')
    parser.add_argument('--lr', default=0.001, type=float,
                        help='learning rate')
    parser.add_argument('--alpha', default=1e-2, type=float,
                        help='alhpa is a hyperparameter in SDNE')
    parser.add_argument('--beta', default=5., type=float,
                        help='beta is a hyperparameter in SDNE')
    parser.add_argument('--nu1', default=1e-5, type=float,
                        help='nu1 is a hyperparameter in SDNE')
    parser.add_argument('--nu2', default=1e-4, type=float,
                        help='nu2 is a hyperparameter in SDNE')
    parser.add_argument('--bs', default=100, type=int,
                        help='batch size of SDNE')
    parser.add_argument('--nhid0', default=1000, type=int,
                        help='The first dim')
    parser.add_argument('--nhid1', default=128, type=int,
                        help='The second dim')
    parser.add_argument('--step_size', default=10, type=int,
                        help='The step size for lr')
    parser.add_argument('--gamma', default=0.9, type=int,
                        help='The gamma for lr')
    args = parser.parse_args()
    return args


# 1
if __name__ == '__main__':
    disease = "Diabetes"  # 正在跑LD+net
    num_epochskf = 500
    num_epochs = 500
    args = parse_args()
    Po_pairs = load("/Datasets/"+disease + "/Positive_pairs.txt")
    Ne_pairs = load("/Datasets/"+disease + "/Negative_pairs.txt")

    train_data = loadtt("/Datasets/" + disease + "/train.txt")#31794
    test_data = loadtt("/Datasets/" + disease + "/test.txt")#7952

    save_figdata = [[i, test_data[i][2]] for i in range(len(test_data))]

    # 2
    # Keep the proteins involved
    proteinID1 = [Po_pairs[i][0] for i in range(len(Po_pairs))]
    proteinID2 = [Po_pairs[i][1] for i in range(len(Po_pairs))]
    proteinID3 = [Ne_pairs[i][0] for i in range(len(Ne_pairs))]
    proteinID4 = [Ne_pairs[i][1] for i in range(len(Ne_pairs))]
    proteinID = list(set(proteinID1 + proteinID2 + proteinID3 + proteinID4))  
    add_nodes = list(set(proteinID3 + proteinID4) - set(proteinID1 + proteinID1))  

    # Get the sequence of each protein and store it in the dictionary
    proteinSeq = []
    for ID in proteinID:
        Seq = readSeq(ID)
        proteinSeq.append(Seq)
    protein = dict(zip(proteinID, proteinSeq))  # Such as {ID: Seq}

    # Construct adjacency matrix and similarity matrix
    Train_File_g = "/Datasets/" + disease + "/train.txt"
    Test_File_g = "/Datasets/"+ disease + "/test.txt"


    Adjsi, nodedic, nodelist = Init22(Test_File_g, Train_File_g, proteinID) 
    Nodecount = len(nodelist)
    # save_figdata = [[i, test_data[i][2]] for i in range(len(test_data))]
    for i in range(len(train_data)):
        temp = [Adjsi[nodelist.index(train_data[i][0])], Adjsi[nodelist.index(train_data[i][1])]]
        train_data[i].append(np.array(temp))
    for i in range(len(test_data)):
        test_data[i].append(np.array([Adjsi[nodelist.index(test_data[i][0])], Adjsi[nodelist.index(test_data[i][1])]]))

    seqfeature=get_bert(protein)

    start = time.clock()

    Indd_train(ModelCon, train_data, test_data, seqfeature)


    end = time.clock()
    print(str(end - start))

