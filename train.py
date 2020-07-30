import torch
from collections import OrderedDict

def batch_train(src_train, trg_train, src_test, trg_test, batch_size, params, regressor, criterion, optimizer):
    batch_loss = 0
    for j in range(batch_size):
        x_tr, y_tr, x_test, y_test = src_train[j], trg_train[j], src_test[j], trg_test[j] # [K, 1], [K, 1], [Q, 1], [Q, 1]
        x_tr.requires_grad_(True)

        preds = regressor(x_tr, params) # [K, 1]
        train_loss = criterion(preds, y_tr)
        
        grads = torch.autograd.grad(train_loss, params.values(), retain_graph = True, create_graph = True)
        
        params = OrderedDict((name, param - 0.01 * grad) for ((name, param), grad) in zip(params.items(), grads))

        preds = regressor(x_test, params)
        batch_loss += criterion(preds, y_test)

    batch_loss /= batch_size

    optimizer.zero_grad()
    batch_loss.backward()
    optimizer.step()

    return batch_loss.item()