import numpy as np
import torch
from collections import OrderedDict
import matplotlib.pyplot as plt

def batch_test(iter, src_test, trg_test, amp, phase, batch_size, params, regressor, device, criterion, num_grad_steps):
    pre_loss, post_loss = 0, 0

    for j in range(batch_size):
        #############################################################
        # plot
        t = np.arange(-5.0, 5.0, 0.01)
        gt = amp[j].numpy() * np.sin(t + phase[j].numpy())
        
        t = torch.tensor(t, dtype=torch.float).unsqueeze(-1).to(device)
        pre = regressor(t, params)
        #############################################################

        x_test, y_test = src_test[j], trg_test[j] # [Q, 1], [Q, 1]
        x_test.requires_grad_(True)

        preds = regressor(x_test, params)
        pre_loss += criterion(preds, y_test)

        for k in range(num_grad_steps):
            preds = regressor(x_test, params) # [K, 1]
            train_loss = criterion(preds, y_test)
            
            grads = torch.autograd.grad(train_loss, params.values(), retain_graph = True, create_graph = True)
            
            params = OrderedDict((name, param - 0.01 * grad) for ((name, param), grad) in zip(params.items(), grads))

        preds = regressor(x_test, params)
        post_loss += criterion(preds, y_test)

        #############################################################
        # plot
        post = regressor(t, params)
        fig, ax = plt.subplots()
        t = t.data.cpu().numpy()
        ax.plot(t, gt, 'r', label = 'ground truth')
        ax.plot(t, pre.data.cpu().numpy(), label = 'pre update')
        ax.plot(t, post.data.cpu().numpy(), label = '1 grad step')
        ax.legend()

        ax.set(title='MAML, K = 10, error : %.3f->%.3f'%(pre_loss, post_loss))
        ax.grid()

        fig.savefig("images/test_%d_iter%d.png"%(j+1, iter))
        plt.close()
        #############################################################

    pre_loss /= batch_size
    post_loss /= batch_size

    return pre_loss.item(), post_loss.item()