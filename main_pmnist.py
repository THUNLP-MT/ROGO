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

## Define MLP model
class MLPNet(nn.Module):
    def __init__(self, n_hidden=100, n_outputs=10):
        super(MLPNet, self).__init__()
        self.act=OrderedDict()
        self.lin1 = Linear(784,n_hidden,bias=False)
        self.lin2 = Linear(n_hidden,n_hidden, bias=False)
        self.fc1  = Linear(n_hidden, n_outputs, bias=False)
        
    def forward(self, x, space = [None, None, None]):
        self.act['Lin1']=x
        x = self.lin1(x, space=space[0])        
        x = F.relu(x)
        self.act['Lin2']=x 
        x = self.lin2(x, space=space[1])        
        x = F.relu(x)
        self.act['fc1']=x
        x = self.fc1(x,space=space[2])
        return x

    def consolidate(self, space=[None, None, None]):
        self.lin1.consolidate(space=space[0])
        self.lin2.consolidate(space=space[1])
        self.fc1.consolidate(space=space[2])
        
def get_model(model):
    return deepcopy(model.state_dict())

def set_model_(model,state_dict):
    model.load_state_dict(deepcopy(state_dict))
    return

def train (args, model, device, x, y, optimizer,criterion):
    model.train()
    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).to(device)
    # Loop batches
    for i in range(0,len(r),args.batch_size_train):
        if i+args.batch_size_train<=len(r): b=r[i:i+args.batch_size_train]
        else: b=r[i:]
        data = x[b].view(-1,28*28)
        data, target = data.to(device), y[b].to(device)
        optimizer.zero_grad()        
        output = model(data)
        loss = criterion(output, target)        
        loss.backward()
        optimizer.step()

def train_projected (args, model,device,x,y,optimizer,criterion,feature_mat, space=[None, None, None]):
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
        data = x[b].view(-1,28*28)
        data, target = data.to(device), y[b].to(device)
        optimizer.zero_grad()        
        output = model(data, space=space)
        loss = criterion(output, target)    
        
        ly = 0
        for k, (m,params) in enumerate(model.named_parameters()):
            if 'scale' in m:
                if space[ly] is not None:
                    penalty = (params - identical_mat[ly]) ** 2
                    loss += penalty.sum() * args.weight
                ly += 1

        loss.backward()        
        # Gradient Projections 
        ki = 0
        for k, (m,params) in enumerate(model.named_parameters()):
            if 'weight' in m:
                sz =  params.grad.data.size(0)
                params.grad.data = params.grad.data - torch.mm(params.grad.data.view(sz,-1),\
                                                    feature_mat[ki]).view(params.size())
                ki += 1
        optimizer.step()


def test (args, model, device, x, y, criterion, space = [None, None, None]):
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
            data = x[b].view(-1,28*28)
            data, target = data.to(device), y[b].to(device)
            output = model(data, space=space)
            loss = criterion(output, target)
            pred = output.argmax(dim=1, keepdim=True) 
            
            correct    += pred.eq(target.view_as(pred)).sum().item()
            total_loss += loss.data.cpu().numpy().item()*len(b)
            total_num  += len(b)

    acc = 100. * correct / total_num
    final_loss = total_loss / total_num
    return final_loss, acc


def get_representation_matrix (net, device, x, y=None): 
    # Collect activations by forward pass
    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).to(device)
    b=r[0:300] # Take random training samples
    example_data = x[b].view(-1,28*28)
    example_data = example_data.to(device)
    example_out  = net(example_data)
    
    batch_list=[300,300,300] 
    mat_list=[] # list contains representation matrix of each layer
    act_key=list(net.act.keys())

    for i in range(len(act_key)):
        bsz=batch_list[i]
        act = net.act[act_key[i]].detach().cpu().numpy()
        activation = act[0:bsz].transpose()
        mat_list.append(activation)

    logging.info('-'*30)
    logging.info('Representation Matrix')
    logging.info('-'*30)
    for i in range(len(mat_list)):
        logging.info ('Layer {} : {}'.format(i+1,mat_list[i].shape))
    logging.info('-'*30)
    return mat_list    


def update_GPM (model, mat_list, threshold, feature_list=[],):
    logging.info ('Threshold: ', threshold) 
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
    thresholds = [0.97, 0.99, 0.99]
    space_thresholds = [0.9, 0.9, 0.9]

    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).to(device)
    b=r[0:125]
    example_data = x[b].view(-1,28*28).to(device)
    target = y[b].to(device)
    
    grad_list=[]
    optimizer.zero_grad()  
    example_out = net(example_data, space=space)
    loss = criterion(example_out, target)         
    loss.backward()  
    for k, (m,params) in enumerate(net.named_parameters()):
        if 'weight' in m:
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
    ## Load PMNIST DATASET
    from dataloader import pmnist as pmd
    data,taskcla,inputsize=pmd.get(seed=args.seed, pc_valid=args.pc_valid)

    acc_matrix=np.zeros((10,10))
    criterion = torch.nn.CrossEntropyLoss()

    task_id = 0
    task_list = []
    for k,ncla in taskcla:
        # specify threshold hyperparameter
        threshold = np.array([0.95,0.99,0.99]) 
     
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
        logging.info ('-'*40)
        logging.info ('Task ID :{} | Learning Rate : {}'.format(task_id, lr))
        logging.info ('-'*40)
        
        if task_id==0:
            model = MLPNet(args.n_hidden, args.n_outputs).to(device)
            logging.info ('Model parameters ---')
            for k_t, (m, param) in enumerate(model.named_parameters()):
                logging.info (str(k_t) + '\t' + str(m) + '\t' + str(param.shape))
            logging.info ('-'*40)

            feature_list =[]
            optimizer = optim.SGD(model.parameters(), lr=lr)
            for epoch in range(1, args.n_epochs+1):
                # Train
                clock0=time.time()
                train(args, model, device, xtrain, ytrain, optimizer, criterion)
                clock1=time.time()
                tr_loss,tr_acc = test(args, model, device, xtrain, ytrain,  criterion)
                logging.info('Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% | time={:5.1f}ms |'.format(epoch,\
                                                            tr_loss,tr_acc, 1000*(clock1-clock0)))
                # Validate
                valid_loss,valid_acc = test(args, model, device, xvalid, yvalid,  criterion)
                logging.info(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss, valid_acc))
                logging.info('\n')
            # Test
            logging.info ('-'*40)
            test_loss, test_acc = test(args, model, device, xtest, ytest,  criterion)
            logging.info('Test: loss={:.3f} , acc={:5.1f}%'.format(test_loss,test_acc))
            # Memory Update  
            mat_list = get_representation_matrix (model, device, xtrain, ytrain)
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

            space = [None, None, None]
            rest = [torch.Tensor(f).to(device) for f in feature_list]
            count = 0
            up = True

            logging.info ('-'*40)
            for epoch in range(1, args.n_epochs+1):
                # Train 
                clock0=time.time()
                train_projected(args, model,device,xtrain, ytrain,optimizer,criterion,feature_mat, space = space)
                clock1=time.time()
                tr_loss, tr_acc = test(args, model, device, xtrain, ytrain,  criterion, space = space)
                logging.info('Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% | time={:5.1f}ms |'.format(epoch,\
                                                        tr_loss, tr_acc, 1000*(clock1-clock0)))
                # Validate
                valid_loss,valid_acc = test(args, model, device, xvalid, yvalid,  criterion, space = space)
                logging.info(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss, valid_acc))
                logging.info('\n')

                if up == True and rest is not None:
                    rest, up, space = update_space(model, xtrain, ytrain, task_id, device, optimizer, criterion, rest, space)

                    if up == True:
                        lr /= args.lr_factor
                        for param_group in optimizer.param_groups:
                            param_group['lr'] /= args.lr_factor

                        count += 1

            model.consolidate(space=space)
            space = [None, None, None]

            for k_t, (m, params) in enumerate(model.named_parameters()):
                if 'scale' in m:
                    mask = torch.eye(params.size(0), params.size(1)).to(device)
                    params.data = mask

            # Test 
            test_loss, test_acc = test(args, model, device, xtest, ytest,  criterion, space = space)
            logging.info('Test: loss={:.3f} , acc={:5.1f}%'.format(test_loss,test_acc))  

            # Memory Update 
            mat_list = get_representation_matrix (model, device, xtrain, ytrain)
            feature_list = update_GPM (model, mat_list, threshold, feature_list)

        # save accuracy 
        jj = 0 
        for ii in np.array(task_list)[0:task_id+1]:
            xtest =data[ii]['test']['x']
            ytest =data[ii]['test']['y'] 
            space = [None, None, None]
            _, acc_matrix[task_id,jj] = test(args, model, device, xtest, ytest,criterion, space = space) 
            jj +=1
        logging.info('Accuracies =')
        for i_a in range(task_id+1):
            logging.info('\t')
            for j_a in range(acc_matrix.shape[1]):
                logging.info('{:5.1f}% '.format(acc_matrix[i_a,j_a]))
            logging.info('\n')
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
    parser = argparse.ArgumentParser(description='PMNIST with ROGO')
    parser.add_argument('--batch_size_train', type=int, default=10, metavar='N',
                        help='input batch size for training (default: 10)')
    parser.add_argument('--batch_size_test', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--n_epochs', type=int, default=5, metavar='N',
                        help='number of training epochs/task (default: 5)')
    parser.add_argument('--seed', type=int, default=2, metavar='S',
                        help='random seed (default: 2)')
    parser.add_argument('--pc_valid',default=0.1,type=float,
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
    # Architecture
    parser.add_argument('--n_hidden', type=int, default=100, metavar='NH',
                        help='number of hidden units in MLP (default: 100)')
    parser.add_argument('--n_outputs', type=int, default=10, metavar='NO',
                        help='number of output units in MLP (default: 10)')
    parser.add_argument('--n_tasks', type=int, default=10, metavar='NT',
                        help='number of tasks (default: 10)')
    parser.add_argument('--weight', type=float, default=1, metavar='W',
                        help='weight for regularization (\beta) (default: 1)')

    parser.add_argument('--savename', type=str, default='save/PMNIST/',
                        help='save path')
    parser.add_argument('--log_path', type=str, default='save/PMNIST/train.log',
                        help='log path')

    args = parser.parse_args()

    if not os.path.exists(args.savename):
       os.makedirs(args.savename)
    if args.log_path:
        log_path = args.log_path.split('.')[0] + '_seed=' + str(args.seed) + '.log'
        set_logger(log_path)
    
    logging.info('='*100)
    logging.info('Arguments =')
    for arg in vars(args):
        logging.info('\t'+str(arg)+':'+str(getattr(args,arg)))
    logging.info('='*100)

    main(args)
