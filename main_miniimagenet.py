### This file is modified from : https://github.com/sahagobinda/GPM

# Copyright (c) THUNLP, Tsinghua University. All rights reserved. 
# # See LICENSE file in the project root for license information.

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import relu, avg_pool2d

import sys
import os
import os.path
from collections import OrderedDict

import logging
import numpy as np
import random
import argparse,time
from copy import deepcopy
from layers import Conv2d, Linear

## Define ResNet18 model
def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

def conv3x3(in_planes, out_planes, stride=1):
    return Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
def conv7x7(in_planes, out_planes, stride=1):
    return Conv2d(in_planes, out_planes, kernel_size=7, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, track_running_stats=False)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, track_running_stats=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes, track_running_stats=False)
            )
        self.act = OrderedDict()
        self.count = 0

    def forward(self, x, space = [None, None]):
        self.count = self.count % 2 
        self.act['conv_{}'.format(self.count)] = x
        self.count +=1
        out = relu(self.bn1(self.conv1(x, space=space[0])))
        self.count = self.count % 2 
        self.act['conv_{}'.format(self.count)] = out
        self.count +=1
        out = self.bn2(self.conv2(out, space=space[1]))
        out += self.shortcut(x)
        out = relu(out)
        return out
    
    def consolidate(self, space = [None, None]):
        self.conv1.consolidate(space=space[0])
        self.conv2.consolidate(space=space[1])

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, taskcla, nf):
        super(ResNet, self).__init__()
        self.in_planes = nf
        self.conv1 = conv3x3(3, nf * 1, 2)
        self.bn1 = nn.BatchNorm2d(nf * 1, track_running_stats=False)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        
        self.taskcla = taskcla
        self.linear=torch.nn.ModuleList()
        for t, n in self.taskcla:
            self.linear.append(nn.Linear(nf * 8 * block.expansion * 9, n, bias=False))
        self.act = OrderedDict()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, space = [None]*20):
        
        bsz = x.size(0)
        self.act['conv_in'] = x.view(bsz, 3, 84, 84)
        out = relu(self.bn1(self.conv1(x.view(bsz, 3, 84, 84), space=space[0]))) 

        out = self.layer1[0](out, space=space[1:3])
        out = self.layer1[1](out, space=space[3:5])
        out = self.layer2[0](out, space=space[5:8])
        out = self.layer2[1](out, space=space[8:10])
        out = self.layer3[0](out, space=space[10:13])
        out = self.layer3[1](out, space=space[13:15])
        out = self.layer4[0](out, space=space[15:18])
        out = self.layer4[1](out, space=space[18:20])

        out = avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        y=[]
        for t,i in self.taskcla:
            y.append(self.linear[t](out))
    
        return y
    
    def consolidate(self, space=[None]*20):
        self.conv1.consolidate(space=space[0])
        self.layer1[0].consolidate(space=space[1:3])
        self.layer1[1].consolidate(space=space[3:5])
        self.layer2[0].consolidate(space=space[5:8])
        self.layer2[1].consolidate(space=space[8:10])
        self.layer3[0].consolidate(space=space[10:13])
        self.layer3[1].consolidate(space=space[13:15])
        self.layer4[0].consolidate(space=space[15:18])
        self.layer4[1].consolidate(space=space[18:20])

def ResNet18(taskcla, nf=32):
    return ResNet(BasicBlock, [2, 2, 2, 2], taskcla, nf)

def get_model(model):
    return deepcopy(model.state_dict())

def set_model_(model,state_dict):
    model.load_state_dict(deepcopy(state_dict))
    return

def adjust_learning_rate(optimizer, epoch, args):
    for param_group in optimizer.param_groups:
        if (epoch ==1):
            param_group['lr']=args.lr
        else:
            param_group['lr'] /= args.lr_factor  

def train(args, model, device, x,y, optimizer,criterion, task_id):
    model.train()
    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).to(device)
    # Loop batches
    for i in range(0,len(r),args.batch_size_train):
        if i+args.batch_size_train<=len(r): b=r[i:i+args.batch_size_train]
        else: b=r[i:]
        data = x[b]
        data, target = data.to(device), y[b].to(device)
        optimizer.zero_grad()        
        output = model(data)
        loss = criterion(output[task_id], target)        
        loss.backward()
        optimizer.step()

def train_projected(args,model,device,x,y,optimizer,criterion,feature_mat,task_id, space=[None]*20):
    model.train()
    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).to(device)

    identical_mat = []
    for k, (m,params) in enumerate(model.named_parameters()):
        if 'scale' in m:
            identical_mat.append(torch.eye(params.size(0)).to(device))

    # Loop batches
    for i in range(0,len(r),args.batch_size_train):
        if i+args.batch_size_train<=len(r): b=r[i:i+args.batch_size_train]
        else: b=r[i:]
        data = x[b]
        data, target = data.to(device), y[b].to(device)
        optimizer.zero_grad()        
        output = model(data, space=space)
        loss = criterion(output[task_id], target)  

        ly = 0
        for k, (m,params) in enumerate(model.named_parameters()):
            if 'scale' in m:
                if space[ly] is not None:
                    penalty = (params - identical_mat[ly]) ** 2
                    loss += penalty.sum() * args.weight
                ly += 1

        loss.backward()
        # Gradient Projections 
        kk = 0 
        for k, (m,params) in enumerate(model.named_parameters()):
            if len(params.size())==4:
                sz =  params.grad.data.size(0)
                params.grad.data = params.grad.data - torch.mm(params.grad.data.view(sz,-1),\
                                                    feature_mat[kk]).view(params.size())
                kk+=1
            elif len(params.size())==1 and task_id !=0:
                params.grad.data.fill_(0)

        optimizer.step()

def test(args, model, device, x, y, criterion, task_id, space=[None]*20):
    model.eval()
    total_loss = 0
    total_num = 0 
    correct = 0
    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).to(device)
    with torch.no_grad():
        # Loop batches
        for i in range(0,len(r),args.batch_size_test):
            if i+args.batch_size_test<=len(r): b=r[i:i+args.batch_size_test]
            else: b=r[i:]
            data = x[b]
            data, target = data.to(device), y[b].to(device)
            output = model(data, space=space)
            loss = criterion(output[task_id], target)
            pred = output[task_id].argmax(dim=1, keepdim=True) 
            
            correct    += pred.eq(target.view_as(pred)).sum().item()
            total_loss += loss.data.cpu().numpy().item()*len(b)
            total_num  += len(b)

    acc = 100. * correct / total_num
    final_loss = total_loss / total_num
    return final_loss, acc

def get_representation_matrix_ResNet18 (net, device, criterion, task_id, x, y=None): 
    # Collect activations by forward pass
    net.eval()
    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).to(device)
    b=r[0:100] # ns=100 examples 
    example_data = x[b]
    example_data = example_data.to(device)
    example_out  = net(example_data)
    
    target = y[b].to(device)
    loss = criterion(example_out[task_id], target)
    loss.backward()

    mat_list = []
    for k, (m,params) in enumerate(net.named_parameters()):
        if len(params.size())==4:
            sz =  params.grad.data.size(0)
            mat = deepcopy(params.grad.data).view(sz,-1)
            mat = mat.cpu().numpy()
            mat_list.append(mat.transpose())
    net.zero_grad()

    print('-'*30)
    print('Representation Matrix')
    print('-'*30)
    for i in range(len(mat_list)):
        print ('Layer {} : {}'.format(i+1,mat_list[i].shape))
    print('-'*30)
    return mat_list  

def update_GPM (model, mat_list, threshold, feature_list=[],):
    logging.info ('Threshold: ' + str(threshold)) 
    if not feature_list:
        # After First Task 
        for i in range(len(mat_list)):
            activation = mat_list[i]
            U,S,Vh = np.linalg.svd(activation, full_matrices=False)
            # criteria (Eq-5)
            sval_total = (S**2).sum()
            sval_ratio = (S**2)/sval_total
            r = np.sum(np.cumsum(sval_ratio)<threshold[i]) #+1  
            feature_list.append(U[:,0:r])
    else:
        for i in range(len(mat_list)):
            activation = mat_list[i]
            U1,S1,Vh1=np.linalg.svd(activation, full_matrices=False)
            sval_total = (S1**2).sum()
            # Projected Representation (Eq-8)
            act_hat = activation - np.dot(np.dot(feature_list[i],feature_list[i].transpose()),activation)
            U,S,Vh = np.linalg.svd(act_hat, full_matrices=False)
            # criteria (Eq-9)
            sval_hat = (S**2).sum()
            sval_ratio = (S**2)/sval_total               
            accumulated_sval = (sval_total-sval_hat)/sval_total
            
            r = 0
            for ii in range (sval_ratio.shape[0]):
                if accumulated_sval < threshold[i]:
                    accumulated_sval += sval_ratio[ii]
                    r += 1
                else:
                    break
            if r == 0:
                logging.info ('Skip Updating GPM for layer: {}'.format(i+1)) 
                continue
            # update GPM
            Ui=np.hstack((feature_list[i],U[:,0:r]))  
            if Ui.shape[1] > Ui.shape[0] :
                feature_list[i]=Ui[:,0:Ui.shape[0]]
            else:
                feature_list[i]=Ui
    
    logging.info('-'*40)
    logging.info('Gradient Constraints Summary')
    logging.info('-'*40)
    for i in range(len(feature_list)):
        logging.info ('Layer {} : {}/{}'.format(i+1,feature_list[i].shape[1], feature_list[i].shape[0]))
    logging.info('-'*40)
    return feature_list 

def update_space(net, x, y, task_id, device, optimizer, criterion, rest_space=None, space=None):

    thresholds = [0.97]*20
    space_thresholds = [0.95]*20

    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).to(device)
    b=r[0:100*5]
    example_data = x[b].to(device)
    target = y[b].to(device)
    
    grad_list=[]
    optimizer.zero_grad()  
    example_out  = net(example_data, space=space)
    loss = criterion(example_out[task_id], target)         
    loss.backward()  
    k_conv = 0
    for k, (m,params) in enumerate(net.named_parameters()):
        if len(params.shape) == 4 and 'weight' in m:
            grad = params.grad.data.detach()
            gr = grad.view(grad.size(0), -1)
            grad_list.append(gr)
            k_conv += 1

    rest = []
    up = False
    for i in range(len(grad_list)):
        frozen_space = deepcopy(rest_space[i])
        current_grad = grad_list[i].transpose(0,1)
        logging.info (f'Frozen Space Size : {frozen_space.size(0)}, {frozen_space.size(1)}')

        U,S,Vh = torch.linalg.svd(current_grad, full_matrices=False)
        sval_total = (S**2).sum()
        sval_ratio = (S**2)/sval_total
        r = 1
        while torch.sum(sval_ratio[:r]) < thresholds[i]:
            r += 1
        U = U[:,0:r]
        logging.info (f'Compress Representation Size ({current_grad.size(0)}, {current_grad.size(1)}) to ({U.size(0)}, {U.size(1)})')

        threshold = space_thresholds[i]
        trusts = []
        importance = 0
        UU = torch.mm(U, U.transpose(0,1))
        while importance < threshold:
            representation = torch.mm(frozen_space.transpose(0,1), torch.mm(UU, frozen_space))
            try:
                Ux,Sx,Vhx = torch.linalg.svd(representation, full_matrices=False)
                x = Ux[:, 0:1]
            except:
                Ux,Sx,Vhx = np.linalg.svd(representation.cpu().numpy(), full_matrices=False)
                x = torch.Tensor(Ux[:, 0:1]).to(device)
            if torch.sum(x) == 0: break
            u = torch.mm(frozen_space, x)
            u /= torch.linalg.norm(u)

            replace = False
            for idx in range(len(x)):
                if x[idx] != 0:
                    if idx > 0 and idx < len(x) - 1:
                        frozen_space = torch.cat([u, frozen_space[:, :idx], frozen_space[:, idx+1:]], dim=1)
                    elif idx == 0:
                        frozen_space = torch.cat([u, frozen_space[:, 1:]], dim=1)
                    else:
                        frozen_space = torch.cat([u, frozen_space[:, :idx]], dim=1)
                    replace = True
                    break
            assert replace == True

            q, _ = torch.linalg.qr(frozen_space)
            trust = q[:, 0:1]
            projection = torch.mm(UU, trust)
            score = torch.linalg.norm(projection) / torch.linalg.norm(trust)
            if score < threshold: break
            frozen_space = q[:, 1:]
            trusts.append(trust)

        if len(trusts) == 0: common_space = None
        else: 
            common_space = torch.cat(trusts, dim=1)

        if space[i] is None:
            new_space = common_space
            if common_space is None: logging.info ('Keep Relaxing Space as None')
            else: logging.info (f'Initiate Relaxing Space as ({new_space.size(0)}, {new_space.size(1)})')
        else:
            exist_space = space[i]
            if common_space is None:
                new_space = exist_space
                logging.info (f'Keep Relaxing Space as Previous ({new_space.size(0)}, {new_space.size(1)})')
            else:
                new_space = torch.cat((exist_space, common_space), dim=1)
                logging.info (f'Expand Relaxing Space from ({exist_space.size(0)}, {exist_space.size(1)}) to ({new_space.size(0)}, {new_space.size(1)})')
        if common_space is not None: up = True

        if new_space is not None: space[i] = new_space.detach()
        
        rest.append(frozen_space)
    return rest, up, space


def main(args):
    tstart=time.time()
    ## Device Setting 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    ## Load MiniImageNet DATASET
    from dataloader import miniimagenet as data_loader
    dataloader = data_loader.DatasetGen(args)
    taskcla, inputsize = dataloader.taskcla, dataloader.inputsize

    acc_matrix=np.zeros((20,20))
    criterion = torch.nn.CrossEntropyLoss()

    task_id = 0
    task_list = []
    for k,ncla in taskcla:
        # specify threshold hyperparameter
        threshold = np.array([0.985 + k * 0.0003] * 20)
        
        data = dataloader.get(k)

        logging.info('*'*100)
        logging.info('Task {:2d} ({:s})'.format(k,data[k]['name']))
        logging.info('*'*100)
        xtrain=data[k]['train']['x']
        ytrain=data[k]['train']['y']
        xvalid=data[k]['valid']['x']
        yvalid=data[k]['valid']['y']
        xtest =data[k]['test']['x']
        ytest =data[k]['test']['y']
        task_list.append(k)

        lr = args.lr 
        best_loss=np.inf
        logging.info ('-'*40)
        logging.info ('Task ID :{} | Learning Rate : {}'.format(task_id, lr))
        logging.info ('-'*40)
        
        if task_id==0:
            model = ResNet18(taskcla,20).to(device) # base filters: 20
 
            logging.info ('Model parameters ---')
            for k_t, (m, param) in enumerate(model.named_parameters()):
                logging.info (str(k_t) + '\t' + str(m) + '\t' + str(param.shape))
            logging.info ('-'*40)

            best_model=get_model(model)
            feature_list =[]
            optimizer = optim.SGD(model.parameters(), lr=lr)

            for epoch in range(1, args.n_epochs+1):
                # Train
                clock0=time.time()
                train(args, model, device, xtrain, ytrain, optimizer, criterion, k)
                clock1=time.time()
                tr_loss,tr_acc = test(args, model, device, xtrain, ytrain,  criterion, k)
                logging.info('Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% | time={:5.1f}ms |'.format(epoch,\
                                                            tr_loss,tr_acc, 1000*(clock1-clock0)))
                # Validate
                valid_loss,valid_acc = test(args, model, device, xvalid, yvalid,  criterion, k)
                logging.info(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss, valid_acc))
                # Adapt lr
                if valid_loss<best_loss:
                    best_loss=valid_loss
                    best_model=get_model(model)
                    patience=args.lr_patience
                    logging.info(' *')
                else:
                    patience-=1
                    if patience<=0:
                        lr/=args.lr_factor
                        logging.info(' lr={:.1e}'.format(lr))
                        if lr<args.lr_min:
                            logging.info('\t')
                            break
                        patience=args.lr_patience
                        adjust_learning_rate(optimizer, epoch, args)
            set_model_(model,best_model)
            # Test
            logging.info ('-'*40)
            test_loss, test_acc = test(args, model, device, xtest, ytest,  criterion, k)
            logging.info('Test: loss={:.3f} , acc={:5.1f}%'.format(test_loss,test_acc))
            # Memory Update  
            mat_list = get_representation_matrix_ResNet18 (model, device, criterion, k, xtrain, ytrain)
            feature_list = update_GPM (model, mat_list, threshold, feature_list)

        else:
            normal_param = [param for name, param in model.named_parameters() if not 'scale' in name] 
            scale_param = [param for name, param in model.named_parameters() if 'scale' in name]
            optimizer = torch.optim.SGD([{'params': normal_param},{'params': scale_param, 'weight_decay': 0, 'lr':args.lr}],lr=args.lr)

            feature_mat = []
            # Projection Matrix Precomputation
            for i in range(len(feature_list)):
                Uf=torch.Tensor(np.dot(feature_list[i],feature_list[i].transpose())).to(device)
                logging.info('Layer {} - Projection Matrix shape: {}'.format(i+1,Uf.shape))
                feature_mat.append(Uf)
            
            space = [None] * 20
            rest = [torch.Tensor(f).to(device) for f in feature_list]
            count = 0
            up = True

            logging.info ('-'*40)
            for epoch in range(1, args.n_epochs+1):
                # Train 
                clock0=time.time()
                train_projected(args, model,device,xtrain, ytrain,optimizer,criterion,feature_mat,k, space=space)
                clock1=time.time()
                tr_loss, tr_acc = test(args, model, device, xtrain, ytrain,criterion,k, space=space)
                logging.info('Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% | time={:5.1f}ms |'.format(epoch,\
                                                        tr_loss, tr_acc, 1000*(clock1-clock0)))
                # Validate
                valid_loss,valid_acc = test(args, model, device, xvalid, yvalid, criterion,k, space=space)
                logging.info(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss, valid_acc))
                # Adapt lr
                if valid_loss<best_loss:
                    best_loss=valid_loss
                    best_model=get_model(model)
                    patience=args.lr_patience
                    logging.info(' *')
                else:
                    patience-=1
                    if patience<=0:
                        lr/=args.lr_factor
                        logging.info(' lr={:.1e}'.format(lr))
                        if lr<args.lr_min:
                            logging.info('\t')
                            break
                        patience=args.lr_patience
                        adjust_learning_rate(optimizer, epoch, args)

                    if lr < 1e-2 and up == True and count < 2:
                        rest, up, space = update_space(model, xtrain, ytrain, task_id, device, optimizer, criterion, rest, space)

                        if up == True:
                            lr = args.lr
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = args.lr
                            count += 1

                logging.info("")
            
            # rounds.append(count)
            set_model_(model,best_model)
            model.consolidate(space)
            space = [None] * 20

            for k_t, (m, params) in enumerate(model.named_parameters()):
                if 'scale' in m:
                    mask = torch.eye(params.size(0), params.size(1)).to(device)
                    params.data = mask

            test_loss, test_acc = test(args, model, device, xtest, ytest,  criterion,k, space=space)
            logging.info('Test: loss={:.3f} , acc={:5.1f}%'.format(test_loss,test_acc))  

            # Memory Update 
            mat_list = get_representation_matrix_ResNet18 (model, device, criterion, k, xtrain, ytrain)
            feature_list = update_GPM (model, mat_list, threshold, feature_list)
                
        # save accuracy 
        jj = 0 
        for ii in np.array(task_list)[0:task_id+1]:
            xtest =data[ii]['test']['x']
            ytest =data[ii]['test']['y'] 
            space = [None]*20
            _, acc_matrix[task_id,jj] = test(args, model, device, xtest, ytest,criterion,ii, space=space) 
            jj +=1
        logging.info('Accuracies =')
        for i_a in range(task_id+1):
            logging.info('\t')
            for j_a in range(acc_matrix.shape[1]):
                logging.info('{:5.1f}% '.format(acc_matrix[i_a,j_a]))
            logging.info('\t')
        # update task id 
        task_id +=1


    logging.info('-'*50)
    logging.info ('Task Order : {}'.format(np.array(task_list)))
    logging.info ('Final Avg Accuracy: {:5.2f}%'.format(acc_matrix[-1].mean())) 
    bwt=np.mean((acc_matrix[-1]-np.diag(acc_matrix))[:-1]) 
    logging.info ('Backward transfer: {:5.2f}%'.format(bwt))
    omega = np.mean(np.diag(acc_matrix)[1:])
    logging.info ('Forward Knowledge Transfer (\Omega_{new}): {:5.2f}%'.format(omega))
    logging.info('[Elapsed time = {:.1f} ms]'.format((time.time()-tstart)*1000))
    logging.info('-'*50)

def set_logger(filepath):
    global logger
    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(filepath)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    _format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(_format)
    ch.setFormatter(_format)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return


if __name__ == "__main__":
    # Training parameters
    parser = argparse.ArgumentParser(description='MiniImageNet with ROGO')
    parser.add_argument('--batch_size_train', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--batch_size_test', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--n_epochs', type=int, default=200, metavar='N',
                        help='number of training epochs/task (default: 200)')
    parser.add_argument('--seed', type=int, default=37, metavar='S',
                        help='random seed (default: 37)')
    parser.add_argument('--pc_valid',default=0.02,type=float,
                        help='fraction of training data used for validation')
    # Optimizer parameters
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--lr_min', type=float, default=1e-3, metavar='LRM',
                        help='minimum lr rate (default: 1e-5)')
    parser.add_argument('--lr_patience', type=int, default=6, metavar='LRP',
                        help='hold before decaying lr (default: 6)')
    parser.add_argument('--lr_factor', type=int, default=2, metavar='LRF',
                        help='lr decay factor (default: 2)')
    parser.add_argument('--weight', type=float, default=5, metavar='W',
                        help='weight for regularization (\beta) (default: 5)')

    parser.add_argument('--savename', type=str, default='save/MiniImageNet/',
                        help='save path')
    parser.add_argument('--log_path', type=str, default='save/MiniImageNet/train.log',
                        help='log path')

    args = parser.parse_args()
    if not os.path.exists(args.savename):
       os.makedirs(args.savename)
    if args.log_path:
        log_path = args.log_path
        set_logger(log_path)

    logging.info('='*100)
    logging.info('Arguments =')
    for arg in vars(args):
        logging.info('\t'+arg+': {}'.format(getattr(args,arg)))
    logging.info('='*100)


    logging.info('save path : {}'.format(args.savename))
    main(args)



