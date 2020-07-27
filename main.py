import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from collections import OrderedDict
from sinusoid import Sinusoid
from model import Regressor

import os, sys, time, argparse

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("results/log.txt", "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass

def save_checkpoint(model, path):
    model_state = {
        'state_dict' : model.state_dict()
    }
    
    torch.save(model_state, path)
    print('A check point has been generated : ' + path)

def main(args):
    device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    
    regressor = Regressor().to(device)
    criterion = nn.MSELoss()
    num_grad_steps = 1

    if not args.test:
        train_dataset = Sinusoid(k_shot=args.k_shot, q_query=args.query, num_tasks=args.num_tasks)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)

        outer_optimizer = torch.optim.Adam(regressor.parameters(), lr = 1e-3)
        
        start = time.time()
        for i, (src_train, trg_train, src_test, trg_test) in enumerate(train_loader):
            # src_train : a tensor with size [batch, K, 1]
            # trg_train : a tensor with size [batch, K, 1]
            # src_test : a tensor with size [batch, Q, 1]
            # trg_test : a tensor with size [batch, Q, 1]

            src_train, trg_train, src_test, trg_test = src_train.to(device), trg_train.to(device), src_test.to(device), trg_test.to(device)
            
            batch_size = src_train.size(0)

            master_params = OrderedDict((name, param) for (name, param) in regressor.named_parameters())
            batch_loss = 0

            # inner loop
            for j in range(batch_size):
                x_tr, y_tr, x_test, y_test = src_train[j], trg_train[j], src_test[j], trg_test[j] # [K, 1], [K, 1], [Q, 1], [Q, 1]
                x_tr.requires_grad_(True)

                preds = regressor(x_tr) # [K, 1]
                train_loss = criterion(preds, y_tr)
                
                grads = torch.autograd.grad(train_loss, regressor.parameters(), retain_graph = True, create_graph = True)
                
                updated_params = OrderedDict((name, param - 0.01 * grad) for ((name, param), grad) in zip(master_params.items(), grads))

                preds = regressor.forward_with_params(x_test, updated_params)
                batch_loss += criterion(preds, y_test)

            batch_loss /= batch_size

            outer_optimizer.zero_grad()
            batch_loss.backward()
            outer_optimizer.step()

            if i % 100 == 0:
                t = time.time() - start
                print('[%d/%d] batch loss : %.4f | time : %.2fs'%(i, len(train_loader), batch_loss.item(), t))
                start = time.time()
            
            # test
            if i % 100 == 0:
                pre_loss, post_loss = 0, 0
                params = OrderedDict((name, param) for (name, param) in regressor.named_parameters())

                for j in range(batch_size):
                    x_test, y_test = src_test[j], trg_test[j] # [Q, 1], [Q, 1]
                    x_test.requires_grad_(True)

                    preds = regressor.forward_with_params(x_test, params)
                    pre_loss += criterion(preds, y_test)

                    for k in range(num_grad_steps):
                        preds = regressor.forward_with_params(x_test, params) # [K, 1]
                        train_loss = criterion(preds, y_test)
                        
                        grads = torch.autograd.grad(train_loss, params.values(), retain_graph = True, create_graph = True)
                        
                        params = OrderedDict((name, param - 0.01 * grad) for ((name, param), grad) in zip(params.items(), grads))

                    preds = regressor.forward_with_params(x_test, params)
                    post_loss += criterion(preds, y_test)

                pre_loss /= batch_size
                post_loss /= batch_size
                print('[%d/%d] mean test loss : %.4f -> %.4f by %d grad steps'%(i, len(train_loader), pre_loss.item(), post_loss.item(), num_grad_steps))
            
            if i % 1000 == 0:
                save_checkpoint(regressor, 'checkpoints/iter_%d'%(i))
    else:
        if os.path.exists(args.checkpoint):
            checkpoint = torch.load(args.checkpoint)
            regressor.load_state_dict(checkpoint['state_dict'])
            print("trained regressor " + args.checkpoint + " is loaded")

        test_dataset = Sinusoid(k_shot=args.k_shot, q_query=args.query, num_tasks=args.num_tasks)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)

        for i, (_, _, src_test, trg_test) in enumerate(test_loader):
            src_test, trg_test = src_test.to(device), trg_test.to(device)
            batch_size = src_test.size(0)

            pre_loss, post_loss = 0, 0
            params = OrderedDict((name, param) for (name, param) in regressor.named_parameters())

            for j in range(batch_size):
                x_test, y_test = src_test[j], trg_test[j] # [Q, 1], [Q, 1]
                x_test.requires_grad_(True)

                preds = regressor.forward_with_params(x_test, params)
                pre_loss += criterion(preds, y_test)

                for k in range(num_grad_steps):
                    preds = regressor.forward_with_params(x_test, params) # [K, 1]
                    train_loss = criterion(preds, y_test)
                    
                    grads = torch.autograd.grad(train_loss, params.values(), retain_graph = True, create_graph = True)
                    
                    params = OrderedDict((name, param - 0.01 * grad) for ((name, param), grad) in zip(params.items(), grads))

                preds = regressor.forward_with_params(x_test, params)
                post_loss += criterion(preds, y_test)

            pre_loss /= batch_size
            post_loss /= batch_size
            print('[%d/%d] mean test loss : %.4f -> %.4f by %d grad steps'%(i, len(train_loader), pre_loss.item(), post_loss.item(), num_grad_steps))    

        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MAML')
    parser.add_argument(
        '--k_shot',
        type=int,
        default=10)

    parser.add_argument(
        '--query',
        type=int,
        default=15)

    parser.add_argument(
        '--num_tasks',
        type=int,
        default=100000)

    parser.add_argument(
        '--batch_size',
        type=int,
        default=10)

    parser.add_argument(
        '--test',
        action='store_true')

    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints/best'
    )

    args = parser.parse_args()
    sys.stdout = Logger()

    main(args)