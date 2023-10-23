### This file is modified from : https://github.com/sahagobinda/GPM

# Copyright (c) THUNLP, Tsinghua University. All rights reserved. 
# # See LICENSE file in the project root for license information.
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
import os.path
from collections import OrderedDict

import logging
import numpy as np
import random
import argparse,time
from copy import deepcopy
from layers import Linear, Conv2d

## Define AlexNet model
def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

class AlexNet(nn.Module):
    def __init__(self,taskcla):
        super(AlexNet, self).__init__()
        self.act=OrderedDict()
        self.map =[]
        self.ksize=[]
        self.in_channel =[]
        self.map.append(32)
        self.conv1 = Conv2d(3, 64, 4, bias=False)
        self.bn1 = nn.BatchNorm2d(64, track_running_stats=False)
        s=compute_conv_output_size(32,4)
        s=s//2
        self.ksize.append(4)
        self.in_channel.append(3)
        self.map.append(s)
        self.conv2 = Conv2d(64, 128, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(128, track_running_stats=False)
        s=compute_conv_output_size(s,3)
        s=s//2
        self.ksize.append(3)
        self.in_channel.append(64)
        self.map.append(s)
        self.conv3 = Conv2d(128, 256, 2, bias=False)
        self.bn3 = nn.BatchNorm2d(256, track_running_stats=False)
        s=compute_conv_output_size(s,2)
        s=s//2
        self.smid=s
        self.ksize.append(2)
        self.in_channel.append(128)
        self.map.append(256*self.smid*self.smid)
        self.maxpool=torch.nn.MaxPool2d(2)
        self.relu=torch.nn.ReLU()
        self.drop1=torch.nn.Dropout(0.2)
        self.drop2=torch.nn.Dropout(0.5)

        self.fc1 = Linear(256*self.smid*self.smid,2048, bias=False)
        self.bn4 = nn.BatchNorm1d(2048, track_running_stats=False)
        self.fc2 = Linear(2048,2048, bias=False)
        self.bn5 = nn.BatchNorm1d(2048, track_running_stats=False)
        self.map.extend([2048])
        
        self.taskcla = taskcla
        self.fc3=torch.nn.ModuleList()
        for t,n in self.taskcla:
            self.fc3.append(torch.nn.Linear(2048,n,bias=False))
        
    def forward(self, x, space=[None, None, None, None, None]):
        bsz = deepcopy(x.size(0))
        self.act['conv1']=x
        x = self.conv1(x, space=space[0])
        x = self.maxpool(self.drop1(self.relu(self.bn1(x))))

        self.act['conv2']=x
        x = self.conv2(x, space=space[1])
        x = self.maxpool(self.drop1(self.relu(self.bn2(x))))

        self.act['conv3']=x
        x = self.conv3(x, space=space[2])
        x = self.maxpool(self.drop2(self.relu(self.bn3(x))))

        x=x.view(bsz,-1)
        self.act['fc1']=x
        x = self.fc1(x, space=space[3])
        x = self.drop2(self.relu(self.bn4(x)))

        self.act['fc2']=x        
        x = self.fc2(x, space=space[4])
        x = self.drop2(self.relu(self.bn5(x)))
        y=[]
        for t,i in self.taskcla:
            y.append(self.fc3[t](x))
            
        return y
    
    def consolidate(self, space = [None, None, None, None, None]):
        self.conv1.consolidate(space=space[0])
        self.conv2.consolidate(space=space[1])
        self.conv3.consolidate(space=space[2])
        self.fc1.consolidate(space=space[3])
        self.fc2.consolidate(space=space[4])

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

def train_projected(args,model,device,x,y,optimizer,criterion,feature_mat,task_id, space=[None,None,None,None,None]):
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
            if k<20 and len(params.size())!=1 and 'weight' in m:
                sz =  params.grad.data.size(0)
                params.grad.data = params.grad.data - torch.mm(params.grad.data.view(sz,-1),\
                                                        feature_mat[kk]).view(params.size())
                kk +=1
            elif (k<20 and len(params.size())==1) and task_id !=0 and 'scale' not in m:
                params.grad.data.fill_(0)

        optimizer.step()

def test(args, model, device, x, y, criterion, task_id, space=[None,None,None,None,None]):
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

def get_representation_matrix (net, device, criterion, task_id, x, y=None): 
    # Collect activations by forward pass
    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).to(device)
    b=r[0:125] # Take 125 random samples 
    example_data = x[b].to(device)
    example_out = net(example_data)

    target = y[b].to(device)
    loss = criterion(example_out[task_id], target)
    loss.backward()

    mat_list = []
    for k, (m,params) in enumerate(net.named_parameters()):
        if k<20 and len(params.size())!=1 and 'weight' in m:
            sz =  params.grad.data.size(0)
            mat = deepcopy(params.grad.data).view(sz,-1)
            mat = mat.cpu().numpy()
            mat_list.append(mat.transpose())
    net.zero_grad()

    logging.info('-'*30)
    logging.info('Representation Matrix')
    logging.info('-'*30)
    for i in range(len(mat_list)):
        logging.info ('Layer {} : {}'.format(i+1,mat_list[i].shape))
    logging.info('-'*30)
    return mat_list    


def update_GPM (model, mat_list, threshold, feature_list=[],):
    logging.info ('Threshold: ' + str(threshold)) 
    if not feature_list:
        # After First Task 
        for i in range(len(mat_list)):
            activation = mat_list[i]
            U,S,Vh = np.linalg.svd(activation, full_matrices=False)
            sval_total = (S**2).sum()
            sval_ratio = (S**2)/sval_total
            r = np.sum(np.cumsum(sval_ratio)<threshold[i]) #+1  
            feature_list.append(U[:,0:r])
    else:
        for i in range(len(mat_list)):
            activation = mat_list[i]
            U1,S1,Vh1=np.linalg.svd(activation, full_matrices=False)
            sval_total = (S1**2).sum()
            act_hat = activation - np.dot(np.dot(feature_list[i],feature_list[i].transpose()),activation)
            U,S,Vh = np.linalg.svd(act_hat, full_matrices=False)
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
    thresholds = [0.97, 0.97, 0.97, 0.99, 0.99]
    space_thresholds = [0.95, 0.95, 0.95, 0.9, 0.9]

    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).to(device)
    b=r[0:125]
    example_data = x[b].to(device)
    target = y[b].to(device)
    
    grad_list=[]
    for i in range(1):
        optimizer.zero_grad()  
        example_out  = net(example_data,space=space)
        loss = criterion(example_out[task_id], target)         
        loss.backward()  

        k_linear = 0
        for k, (m,params) in enumerate(net.named_parameters()):
            if 'weight' in m and 'bn' not in m:
                if len(params.shape) == 4:
                    grad = params.grad.data.detach()
                    gr = grad.view(grad.size(0), -1)
                    grad_list.append(gr)
                else:
                    if 'fc3' in m and k_linear == task_id:
                        grad = params.grad.data.detach()
                        grad_list.append(grad)
                        k_linear += 1
                    elif 'fc3' not in m:
                        grad = params.grad.data.detach()
                        grad_list.append(grad) 

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
    ## Load CIFAR100 DATASET
    from dataloader import cifar100 as cf100
    data,taskcla,inputsize=cf100.get(seed=args.seed, pc_valid=args.pc_valid)

    acc_matrix=np.zeros((10,10))
    criterion = torch.nn.CrossEntropyLoss()

    task_id = 0
    task_list = []
    for k,ncla in taskcla:
        # specify threshold hyperparameter
        threshold = np.array([0.97] * 5) + task_id*np.array([0.003] * 5)
     
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
            model = AlexNet(taskcla).to(device)
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
                logging.info('\t')
            set_model_(model,best_model)
            # Test
            logging.info ('-'*40)
            test_loss, test_acc = test(args, model, device, xtest, ytest,  criterion, k)
            logging.info('Test: loss={:.3f} , acc={:5.1f}%'.format(test_loss,test_acc))
            # Memory Update  
            mat_list = get_representation_matrix (model, device, criterion, k, xtrain, ytrain)
            feature_list = update_GPM (model, mat_list, threshold, feature_list)

        else:            
            normal_param = [param for name, param in model.named_parameters() if not 'scale' in name] 
            scale_param = [param for name, param in model.named_parameters() if 'scale' in name]
            optimizer = torch.optim.SGD([{'params': normal_param},{'params': scale_param, 'weight_decay': 0, 'lr':args.lr}],lr=args.lr)
            
            feature_mat = []
            # Projection Matrix Precomputation
            for i in range(len(model.act)):
                Uf=torch.Tensor(np.dot(feature_list[i],feature_list[i].transpose())).to(device)
                logging.info('Layer {} - Projection Matrix shape: {}'.format(i+1,Uf.shape))
                feature_mat.append(Uf)

            space = [None, None, None, None, None]
            rest = [torch.Tensor(f).to(device) for f in feature_list]
            count = 0
            up = True

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
                        
                        if lr <= 5e-4 and up == True and count < 2:

                            rest, up, space = update_space(model, xtrain, ytrain, task_id, device, optimizer, criterion, rest, space)
                            
                            if up == True:
                                lr = args.lr
                                for param_group in optimizer.param_groups:
                                    param_group['lr'] = args.lr
                                count += 1
                
                logging.info('\t')
            
            # rounds.append(count)
            set_model_(model,best_model)
            model.consolidate(space=space)
            space=[None, None, None, None, None]

            for k_t, (m, params) in enumerate(model.named_parameters()):
                if 'scale' in m:
                    mask = torch.eye(params.size(0), params.size(1)).to(device)
                    params.data = mask

            # Test 
            test_loss, test_acc = test(args, model, device, xtest, ytest,  criterion,k, space=space)
            logging.info('Test: loss={:.3f} , acc={:5.1f}%'.format(test_loss,test_acc))  
                        
            # Memory Update 
            mat_list = get_representation_matrix (model, device, criterion, k, xtrain, ytrain)
            feature_list = update_GPM (model, mat_list, threshold, feature_list)

        # save accuracy 
        jj = 0 
        ids = task_id + 2
        if ids > len(task_list): ids = task_id + 1
        for ii in np.array(task_list)[0:ids]:
            xtest =data[ii]['test']['x']
            ytest =data[ii]['test']['y'] 
            space = [None, None, None, None, None]
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
    parser = argparse.ArgumentParser(description='CIFAR100-Split with ROGO')
    parser.add_argument('--batch_size_train', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--batch_size_test', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--n_epochs', type=int, default=200, metavar='N',
                        help='number of training epochs/task (default: 200)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--pc_valid',default=0.05,type=float,
                        help='fraction of training data used for validation')
    # Optimizer parameters
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--lr_min', type=float, default=1e-5, metavar='LRM',
                        help='minimum lr rate (default: 1e-5)')
    parser.add_argument('--lr_patience', type=int, default=6, metavar='LRP',
                        help='hold before decaying lr (default: 6)')
    parser.add_argument('--lr_factor', type=int, default=2, metavar='LRF',
                        help='lr decay factor (default: 2)')
    parser.add_argument('--weight', type=float, default=1, metavar='W',
                        help='weight for regularization (\beta) (default: 1)')

    parser.add_argument('--savename', type=str, default='save/CIFAR100_split/',
                        help='save path')
    parser.add_argument('--log_path', type=str, default='save/CIFAR100_split/train.log',
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
        logging.info('\t'+str(arg)+':'+str(getattr(args,arg)))
    logging.info('='*100)

    main(args)