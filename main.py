import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from collections import OrderedDict
from sinusoid import Sinusoid
from model import Regressor
from utils import Logger, save_checkpoint
from train import batch_train
from test import batch_test

import os, sys, time, argparse
import numpy as np
import matplotlib.pyplot as plt

def main(args):
    device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    
    regressor = Regressor().to(device)
    criterion = nn.MSELoss()

    if not args.test:
        train_dataset = Sinusoid(k_shot=args.k_shot, q_query=args.query, num_tasks=args.num_tasks)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
        torch.save(train_loader, args.dataloader)

        outer_optimizer = torch.optim.Adam(regressor.parameters(), lr = 1e-3)
        
        start = time.time()
        for i, (src_train, trg_train, src_test, trg_test, (amp, phase)) in enumerate(train_loader):
            # src_train,trg_train : a tensor with size [batch, K, 1]
            # src_test, trg_test : a tensor with size [batch, Q, 1]

            src_train, trg_train, src_test, trg_test = src_train.to(device), trg_train.to(device), src_test.to(device), trg_test.to(device)
            batch_size = src_train.size(0)
            params = OrderedDict((name, param) for (name, param) in regressor.named_parameters())

            # inner loop
            batch_loss = batch_train(src_train, trg_train, src_test, trg_test,
                                    batch_size, params, regressor, criterion, outer_optimizer)

            if i % 100 == 0:
                t = time.time() - start
                print('[%d/%d] batch loss : %.4f | time : %.2fs'%(i, len(train_loader), batch_loss, t))
                start = time.time()
            
            # test
            if i % 10000 == 0:
                params = OrderedDict((name, param) for (name, param) in regressor.named_parameters())

                pre_loss, post_loss = batch_test(i, src_test, trg_test, amp, phase, 
                                                batch_size, params, regressor, device, criterion, args.num_grad_steps)

                print('[%d/%d] mean test loss : %.4f -> %.4f by %d grad steps'
                        %(i+1, len(train_loader), pre_loss, post_loss, args.num_grad_steps))

                save_checkpoint(regressor, 'checkpoints/iter_%d.pt'%(i))
        
        print("Training Finished")
        save_checkpoint(regressor, 'checkpoints/final.pt')

    else:
        if os.path.exists(args.checkpoint) and os.path.exists(args.dataloader):
            data_loader = torch.load(args.dataloader)
            checkpoint = torch.load(args.checkpoint)
            regressor.load_state_dict(checkpoint['state_dict'])
            print("trained regressor " + args.checkpoint + " and dataloader are loaded successfully")
        else:
            raise ValueError("There's no such files or directory")
        
        params = OrderedDict((name, param) for (name, param) in regressor.named_parameters())

        for i, (_, _, src_test, trg_test, (amp, phase)) in enumerate(data_loader):
            if i > args.num_test_batch - 1 :
                break
            
            src_test, trg_test = src_test.to(device), trg_test.to(device)
            batch_size = src_test.size(0)

            pre_loss, post_loss = batch_test(i, src_test, trg_test, amp, phase, 
                                            batch_size, params, regressor, device, criterion, args.num_grad_steps)

            print('[%d/%d] mean test loss : %.4f -> %.4f by %d grad steps'
                    %(i+1, args.num_test_batch, pre_loss, post_loss, args.num_grad_steps))


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
        default=1000000)

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
        default='checkpoints/final.pt')

    parser.add_argument(
        '--num_test_batch',
        type=int,
        default=5)

    parser.add_argument(
        '--num_grad_steps',
        type=int,
        default=1)

    parser.add_argument(
        '--dataloader',
        type=str,
        default='dataloader.pt')

    args = parser.parse_args()
    sys.stdout = Logger()

    main(args)