import torch 
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt


def train(model, graph, dataset, train_iter, opti, device):

    # model_path = os.getcwd() + "/saved_models/" + dataset.lower() + "_" + str(hidden_layers) + "_layers_.pt"
    model_path = 'fittest_model.pt'
    losses = []
    epoch_idx = [i for i in range(train_iter)]
    valid_acc_list = []
    best_val_acc = 0.0

    # training loop
    for epoch in range(train_iter):

        model.train()
        opti.zero_grad()
        src = graph.edges()[0]
        tgt = graph.edges()[1]
        edge_index = torch.zeros(2, src.shape[0], dtype=torch.long).to(device)
        edge_index[0], edge_index[1] = src, tgt
        out = model(graph.ndata['feat'], edge_index)
        out = out.to(device)
        loss = F.cross_entropy(out[graph.ndata['train_mask']], graph.ndata['label'][graph.ndata['train_mask']])
        val_loss = F.cross_entropy(out[graph.ndata['val_mask']], graph.ndata['label'][graph.ndata['val_mask']])
        loss.backward()
        opti.step()
        losses.append(loss.item())

        pred = out.argmax(dim = 1)
        train_acc = (pred[graph.ndata['train_mask']] == graph.ndata['label'][graph.ndata['train_mask']]).float().mean()
        valid_acc = (pred[graph.ndata['val_mask']] == graph.ndata['label'][graph.ndata['val_mask']]).float().mean()
        valid_acc_list.append(valid_acc.item())
        
        if valid_acc > best_val_acc:
            best_val_acc = valid_acc
            torch.save({'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opti.state_dict(),
            'val_acc': valid_acc,
            'train_acc': train_acc,
            'loss': loss},
            model_path)

        if (epoch+1) % 100 == 0:
            print(f"epoch: {epoch + 1:03d} || train loss: {loss:.4f} || train_acc:{train_acc:.4f} || valid_loss: {val_loss:.4f} || valid_acc: {valid_acc:.4f} || best_valid_acc: {best_val_acc:.4f}")

    plt.plot(epoch_idx, losses, color = 'red')
    # plt.plot(epoch_idx, test_acc_list, color = 'green')
    plt.plot(epoch_idx, valid_acc_list, color = 'blue')
    plt.legend(['Loss', 'Validation Accuracy'])
    plt.savefig(os.getcwd() + "/loss_plot_" + dataset + "_.png")

    return model_path