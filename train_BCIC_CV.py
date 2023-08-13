
import time
import pandas as pd
import numpy as np
import random

from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
import torch.utils.data
from Net.model import LGL_BCI
from Net.early_stopping import EarlyStopping
from Net.args import args_parser
import Net.geoopt as geoopt
from Net.utils import DataLoader, dataloader_in_main


SEED = 123
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def adjust_learning_rate(optimizer, epoch):
    optimizer.lr = args.initial_lr * (args.decay ** (epoch // 100))


def main(args, train, test, train_y, test_y, sub, total_sub, kf_iter, validation):

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device   = torch.device("cuda" if use_cuda else "cpu")
            
    train       = Variable(torch.from_numpy(train)).double()
    test        = Variable(torch.from_numpy(test)).double()
    train_y     = Variable(torch.LongTensor(train_y))
    test_y      = Variable(torch.LongTensor(test_y))

    train_dataset = dataloader_in_main(train, train_y)
    test_dataset  = dataloader_in_main(test, test_y)

    train_kwargs = {'batch_size': args.train_batch_size}
    if use_cuda:
          cuda_kwargs ={'num_workers': 1,
                          'pin_memory': True,
                          'shuffle': True
          }
          train_kwargs.update(cuda_kwargs)

    valid_kwargs = {'batch_size': args.valid_batch_size}
    if use_cuda:
          cuda_kwargs ={'num_workers': 1,
                          'pin_memory': True,
                          'shuffle': True
          }
          valid_kwargs.update(cuda_kwargs)

    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
          cuda_kwargs ={'num_workers': 1,
                          'pin_memory': True,
                          'shuffle': True
          }
          test_kwargs.update(cuda_kwargs)

    train_loader  = torch.utils.data.DataLoader(dataset= train_dataset, **train_kwargs)
    test_loader   = torch.utils.data.DataLoader(dataset= test_dataset,  **test_kwargs)


    model = LGL_BCI().to(device)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'The model has {count_parameters(model):,} trainable parameter')

    optimizer1 = geoopt.optim.RiemannianAdam([
        {'params': model.BiMap_Block1.parameters()},
        {'params': model.BiMap_Block2.parameters()}
    ], lr=args.initial_lr)

    optimizer2 = torch.optim.Adam([
                      {'params': model.Temporal_Block.parameters()},
                      {'params': model.Attention.parameters()},
                      {'params': model.Classifier.parameters()}
                  ], lr=args.initial_lr)

    early_stopping = EarlyStopping(
        alg_name = args.alg_name, 
        path_w   = args.weights_folder_path + args.alg_name + '_checkpoint.pt', 
        patience = args.patience, 
        verbose  = True, 
        )

    def eigen_decomp(W, L):
        _, _, _, _no = W.size()
        batch_size, _m, _ni, _ = L.size()
        L = L.detach().numpy()
        eigenvalues, eigenvectors = np.linalg.eig(-L)
        sorted_indices = np.argsort(eigenvalues)[:, :, ::-1]

        indices = torch.tensor(np.ascontiguousarray(sorted_indices))
        index = indices
        eigenvectors = torch.tensor(eigenvectors)
        indices = indices[:, :, :_no].view(batch_size * _m, -1).unsqueeze(1).expand(batch_size * _m, _ni, _no)

        eigenvectors = eigenvectors.view(batch_size * _m, _ni, _ni)
        topk_eigvecs = torch.gather(eigenvectors, -1, indices)
        topk_eigvecs = torch.mean(topk_eigvecs.view(batch_size, _m, _ni, _no), dim=0)

        return topk_eigvecs, index

    best_bal_acc = 0

    print('#####Start Trainning######')

    for epoch in range(1, args.epochs+1):

        adjust_learning_rate(optimizer1, epoch)
        adjust_learning_rate(optimizer2, epoch)

        model.train()

        train_correct = 0
    
        for batch_idx, (batch_train, batch_train_y) in enumerate(train_loader):

            optimizer1.zero_grad()
            optimizer2.zero_grad()

            logits, L = model(batch_train.to(device))

            output = F.log_softmax(logits, dim = -1)
            loss  = F.cross_entropy(output, batch_train_y.to(device))

            loss.backward()
            optimizer1.step()
            optimizer2.step()

            with torch.no_grad():
                aaa, _ = eigen_decomp(model.BiMap_Block2.W.data, L)
                model.BiMap_Block2.W.data.copy_(aaa)

            if batch_idx % args.log_interval == 0:
                print('----#------#-----#-----#-----#-----#-----#-----')
                pred    = output.data.max(1, keepdim=True)[1]
                train_correct += pred.eq(batch_train_y.to(device).data.view_as(pred)).long().cpu().sum()
                torch.save(model.state_dict(), args.weights_folder_path + args.alg_name+'_model.pth')

                print('['+args.alg_name+': Sub No.{}/{} Fold {}/10, Epoch {}/{}, Completed {:.0f}%]:\nTrainning loss {:.10f} Acc.: {:.4f}'.format(\
                        sub, total_sub, kf_iter+1, epoch, args.epochs, 100. * (1+batch_idx) / len(train_loader), loss.cpu().detach().numpy(),\
                        train_correct.item()/len(train_loader.dataset)))

    print('###############################################################')
    print('START TESTING')
    print('###############################################################')

    
    model.eval()

    test_loss    = 0
    test_correct = 0

    with torch.no_grad():
        for batch_idx, (batch_test, batch_test_y) in enumerate(test_loader):

            logits, L = model(batch_test.to(device))
            with torch.no_grad():
                _, index = eigen_decomp(model.BiMap_Block2.W.data, L)
            attention_weight = model.attention_weight
            multi_head = model.multi_head

            label = batch_test_y.view(batch_test_y.size()[0], 1)

            attention_weight = torch.cat((attention_weight, label), dim=-1)


            output        = F.log_softmax(logits, dim = -1)
            loss = F.cross_entropy(output, batch_test_y.to(device))
            test_loss += loss

            
            test_pred     = output.data.max(1, keepdim=True)[1]
            test_correct += test_pred.eq(batch_test_y.to(device).data.view_as(test_pred)).long().cpu().sum()

            print('-----------------------------------')
            print('Testing Batch {}:'.format(batch_idx))
            print(' Pred Label:', test_pred.view(1, test_pred.shape[0]).cpu().numpy()[0])
            print('Ground Truth:', batch_test_y.numpy())

    return test_correct.item()/len(test_loader.dataset), test_loss.item()/len(test_loader.dataset), index, \
           attention_weight, multi_head


if __name__ == '__main__':

    index_list = []
    attention_weight_list = []
    multi_head_list = []

    args   = args_parser()

    alg_df = pd.DataFrame(columns=['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10', 'Avg'])
    alg_df_v = pd.DataFrame(columns=['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10', 'Avg'])

    print('############Start Task#################')
    accuracy_list = []

    for sub in range(args.start_No, args.end_No + 1):

        index_temp = []
        attention_weight_temp = []
        multi_head_temp = []

        loader = DataLoader(dataset='BCIC2a',
                            data_source='E',
                            subject=sub,
                            data_format=None,
                            data_type='time_domain',
                            dataset_path='./datasets')


        alg_record = []

        start      = time.time()

        for kf_iter in range(0, 10):
            X_train, y_train = loader.load_train_set(fold=kf_iter+1)
            X_test, y_test = loader.load_test_set(fold=kf_iter+1)

            acc, loss, index, attention_weight, multi_head = main(
                args       = args,
                train      = X_train,
                test       = X_test,
                train_y    = y_train,
                test_y     = y_test,
                sub        = sub,
                total_sub  = args.end_No - args.start_No + 1,
                kf_iter    = kf_iter,
                validation = True,
                )

            print('##############################################################')

            print(args.alg_name + ' Testing Loss.: {:4f} Acc: {:4f}'.format(loss, acc))

            alg_record.append(acc)

            index_temp.append(index)
            attention_weight_temp.append(attention_weight)
            multi_head_temp.append(multi_head)

        index_temp = torch.cat(index_temp, dim=0)
        attention_weight_temp = torch.cat(attention_weight_temp, dim=0)
        multi_head_temp = torch.cat(multi_head_temp, dim=0)

        index_list.append(index_temp)
        attention_weight_list.append(attention_weight_temp)
        multi_head_list.append(multi_head_temp)

        end = time.time()

        alg_record.append(np.mean(alg_record))
        alg_df.loc[sub] = alg_record

        accuracy_list.append(np.mean(alg_record))

        alg_df.to_csv(args.folder_name + "/" + str(args.epochs) + '.csv', index = False)

    mean_accuracy = np.mean(accuracy_list)
    std_dev_t = np.std(accuracy_list)

    print("test", [mean_accuracy, std_dev_t])





