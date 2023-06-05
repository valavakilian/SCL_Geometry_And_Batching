import torch
from tqdm import tqdm
from scipy.sparse.linalg import svds
import numpy as np
from torchvision import datasets, transforms
from sklearn.cluster import KMeans


from loggers import *





# ------- train fcn ---------------------------------------------------------------------------------------------------
def train(model, criterion, args, device, train_loader, optimizer, epoch, repeated_examples_dataloder, transform_augment_train = None):
    model.train()

    if transform_augment_train == None:
        transform_batch_augment = transforms.Compose([
        transforms.RandomVerticalFlip(p = 1.0)
        ])
        print("Just a flip for augmentation")
    elif args.dataset == "CIFAR10":
        input_ch        = 3
        im_size = 32
        padded_im_size = 32
        train_transforms_augment_list = [transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
        train_transforms_augment_list.insert(0, transforms.Pad((padded_im_size - im_size)//2))
        train_transforms_augment_list.insert(1, transforms.RandomCrop(im_size, padding=4))
        train_transforms_augment_list.insert(2, transforms.RandomHorizontalFlip())
        transform_augment_train = transforms.Compose(train_transforms_augment_list)
        transform_batch_augment = transform_augment_train
    elif args.dataset == "MNIST":
        input_ch        = 1
        im_size = 28
        padded_im_size = 32
        train_transforms_augment_list = [transforms.Normalize(0.1307, 0.3081)]
        train_transforms_augment_list.insert(0, transforms.Pad((padded_im_size - im_size)//2))
        train_transforms_augment_list.insert(1, transforms.RandomCrop(im_size, padding=4))
        train_transforms_augment_list.insert(2, transforms.RandomHorizontalFlip())
        transform_batch_augment = transforms.Compose(train_transforms_augment_list)
        transform_batch_augment = transform_augment_train

    
    repeated_example_data, repeated_example_target = None, None
    if repeated_examples_dataloder is not None:
        for batch_idx, (repeated_data, repeated_target) in enumerate(repeated_examples_dataloder, start=1):
            repeated_example_data, repeated_example_target = repeated_data, repeated_target
            # print(repeated_example_target.shape)
            # input()

    bsz = args.batch_size 

    cls_num_list_train = {}
    for c in range(0, args.K):
        cls_num_list_train[c] = 0

    pbar = tqdm(total=len(train_loader), position=0, leave=True)
    for batch_idx, (data, target) in enumerate(train_loader, start=1):

        if repeated_examples_dataloder is not None:
            data = torch.cat((data, repeated_example_data), 0)
            target = torch.cat((target, repeated_example_target), 0)

        augmented_batch = transform_batch_augment(data)
        data = torch.cat((data, augmented_batch), 0)
        target = torch.cat((target, target), 0)

        if repeated_examples_dataloder is not None:
            if data.shape[0] != 2 * (args.batch_size + len(repeated_example_data)):
                print("I KEEP GETTING RUN")
                continue
        else:
            if data.shape[0] != 2 * args.batch_size:
                print("I KEEP GETTING RUN")
                continue

          
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        if args.loss_type == 'SCL':
            h = model(data)
            h = torch.unsqueeze(h, 1)
        elif args.loss_type == 'CE':
            out = model(data)
        
        if args.loss_type == "SCL":
            loss = criterion(h, target)
        elif args.loss_type == "CE":
            loss = criterion(out, target)

        elif args.loss_type == "CE":
            predicted = torch.argmax(out, dim=1)

        loss.backward()
        optimizer.step()

        if args.loss_type  == 'SCL':
            accuracy = 0
        elif args.loss_type  == 'CE':
            accuracy = torch.mean((predicted == target).float()).item()
            
        pbar.update(1)
        pbar.set_description(
            'Train\t\tEpoch: {} [{}/{} ({:.0f}%)] \t'
            'Batch Loss: {:.6f} \t'
            'Batch Accuracy: {:.6f}'.format(
                epoch,
                batch_idx,
                len(train_loader),
                100. * batch_idx / len(train_loader),
                loss.item(),
                accuracy))

        if args.debug and batch_idx > 20:
            break
        

        for label in target:
            cls_num_list_train[label.item()] += 1
    pbar.close()

    print("cls_num_list_train: " + str(cls_num_list_train))

    return 

# ------- analysis fcn ------------------------------------------------------------------------------------------------
def analysis(logger, model, criterion_summed, args, device, analysis_loader, classifier, epoch, Mu_for_NCC = None):
    model.eval()

    N             = 0
    mean          = [0 for _ in range(args.K)]
    Sw            = 0
    Sw_C = [0 for _ in range(args.K)]

    loss          = 0
    net_accuracy   = 0
    NCC_net_accuracy = 0

    n_c = {}
    per_class_acc = {}
    NCC_per_class_accuracy = {}
    for c in range(0, args.K):
        n_c[c] = 0
        per_class_acc[c] = 0
        NCC_per_class_accuracy[c] = 0 
    
    with torch.no_grad():
        for computation in ['Mean', 'Cov']:
            pbar = tqdm(total=len(analysis_loader), position=0, leave=True)
            for batch_idx, (data, target) in enumerate(analysis_loader, start=1):
                if data.shape[0] != args.batch_size:
                      continue
                
                data, target = data.to(device), target.to(device)

                if args.loss_type == 'SCL':
                    h = model(data)
                elif args.loss_type == 'CE':
                    output = model(data)
                    h = torch.zeros(data.shape[0], data.shape[1])# features.value.data.view(data.shape[0], -1)
                    predicted = torch.argmax(output, dim=1)  

                # during calculation of class means, calculate loss
                if computation == 'Mean':
                    if args.loss_type == 'SCL':
                        loss += criterion_summed(h, target).item()
                    elif args.loss_type == 'CE':
                        loss += criterion_summed(output, target).item()

                for c in range(0, args.K):    

                    # features belonging to class c
                    idxs = (target == c).nonzero(as_tuple=True)[0]

                    # skip if no class-c in this batch
                    if len(idxs) == 0: 
                        continue

                    h_c = h[idxs,:].double() # B CHW

                    if computation == 'Mean':
                        # update class means
                        mean[c] += torch.sum(h_c, dim=0) # CHW
                        n_c[c] += h_c.shape[0]
                        N += h_c.shape[0]

                        if args.loss_type == "CE":
                            # per class classifier accuracy
                            per_class_acc[c] += ((predicted == target) * (target == c)).sum().item()

                    elif computation == 'Cov':
                        # update within-class cov
                        z = h_c - mean[c].unsqueeze(0) # B CHW

                        # for loop - for solving memory issue :((
                        for z_i in range(z.shape[0]):
                            temp = torch.matmul(z[z_i, :].reshape((-1, 1)), z[z_i, :].reshape((1, -1)))
                            Sw += temp
                            Sw_C[c] += temp
                        
                        if args.loss_type == "CE":
                            # per class correct predictions
                            net_pred_for_c = torch.argmax(output[idxs,:], dim=1)
                            net_accuracy += (net_pred_for_c == target[idxs]).sum().item()
                        
                        # 2) agreement between prediction and nearest class center
                        if Mu_for_NCC is None:
                            NCC_scores = torch.stack([torch.norm(h_c[i,:] - M.T,dim=1) \
                                        for i in range(h_c.shape[0])])
                        else:
                            NCC_scores = torch.stack([torch.norm(h_c[i,:] - Mu_for_NCC.T,dim=1) \
                                        for i in range(h_c.shape[0])])
                        NCC_pred = torch.argmin(NCC_scores, dim=1)
                        NCC_net_accuracy += sum(NCC_pred==c).item()
                        NCC_per_class_accuracy[c] += sum(NCC_pred==c).item()

                if args.debug and batch_idx > 20:
                    break

                pbar.update(1)
                pbar.set_description(
                    'Analysis {}\t'
                    'Epoch: {} [{}/{} ({:.0f}%)]'.format(
                        computation,
                        epoch,
                        batch_idx,
                        len(analysis_loader),
                        100. * batch_idx/ len(analysis_loader)))

                if args.debug and batch_idx > 20:
                    break
            pbar.close()

            if computation == 'Mean':
                for c in range(args.K):
                    mean[c] /= n_c[c]
                    M = torch.stack(mean).T
                loss /= N
            elif computation == 'Cov':
                Sw /= N
                for c in range(0, args.K):
                    Sw_C[c] /= n_c[c]
    
        # loss with weight decay
        reg_loss = loss
        for param in model.parameters():
            reg_loss += 0.5 * args.weight_decay * torch.sum(param**2).item()
        

        net_accuracy = net_accuracy / N
        NCC_net_accuracy = NCC_net_accuracy / N
        for c in range(0, args.K):
            per_class_acc[c] /= n_c[c]
            NCC_per_class_accuracy[c] /= n_c[c]
        
        # avg norm
        if args.loss_type == "CE":
            W  = classifier.weight
        else:
            W  = None

        Update_Geometry_Prop(logger, args, loss, reg_loss, M, W, Sw, net_accuracy, per_class_acc, NCC_net_accuracy, NCC_per_class_accuracy, n_c, N)

    return