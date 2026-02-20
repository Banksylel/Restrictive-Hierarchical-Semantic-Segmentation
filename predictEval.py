import torch
import os
import argparse
import time
import numpy as np
import glob
import cv2

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from skimage.io import imread

from Data import dataloaders
from Models import models
from Metrics import performance_metrics
from train import get_classes

import json
import pandas as pd

from train import get_metrics
import csv
import copy

import sys

from config import config
from config import update_config
from collections import deque


# gets map of children nodes
def children_map(tree):
    ch = {}
    stack = [tree]
    while stack:
        t = stack.pop()
        for k, v in t.items():
            if isinstance(v, dict) and len(v) > 0:
                ch[k] = list(v.keys())
                stack.append(v)
            else:
                ch[k] = []
    return ch
# gets breath first order of hierarchy
def bfs_order(tree):
    q = deque(tree.items())
    order = []
    while q:
        name, subtree = q.popleft()
        order.append(name)
        if isinstance(subtree, dict) and len(subtree) > 0:
            for cn, csub in subtree.items():
                q.append((cn, csub))
    return order

# gets the list of classes in order of bfs_order, separated by levels
def levels_bfs(tree):
    levels = []
    q = deque([(name, subtree, 0) for name, subtree in tree.items()])
    while q:
        name, subtree, d = q.popleft()
        if len(levels) <= d:
            levels.append([])
        levels[d].append(name)
        if isinstance(subtree, dict) and len(subtree) > 0:
            for cn, csub in subtree.items():
                q.append((cn, csub, d+1))
    return levels

# gets all decendant classes for a given parent class
def descendant_leaves(node, children, is_leaf):
    if is_leaf[node]:
        return [node]
    leaves = []
    for c in children[node]:
        leaves.extend(descendant_leaves(c, children, is_leaf))
    return leaves

# gets the parent mask as the sum of its children masks
def get_parent_masks(in_out, target, tree, leaf_index):
    X, Y = in_out[0], target[0]
    B, C, H, W = X.shape
    dev = X.device
    dtypeX, dtypeY = X.dtype, Y.dtype
    children   = children_map(tree)
    names_bfs  = bfs_order(tree)
    is_leaf    = {n: len(children[n]) == 0 for n in names_bfs}
    parent_names = [n for n in names_bfs if not is_leaf[n]]
    # Precompute descendant leaves for each parent
    desc = {p: descendant_leaves(p, children, is_leaf) for p in parent_names}
    # Validate leaf_index contains all required leaves and indices are in range
    for p, leaves in desc.items():
        if len(leaves) == 0:
            # This shouldn't happen for a "parent", but guard anyway
            raise ValueError(f"Parent '{p}' has no descendant leaves.")
        bad = [l for l in leaves if l not in leaf_index]
        if bad:
            raise KeyError(f"Missing leaf_index entries for {bad} (needed by parent '{p}').")
        idxs = [leaf_index[l] for l in leaves]
        if min(idxs) < 0 or max(idxs) >= C:
            raise IndexError(
                f"Parent '{p}' has leaf indices out of bounds: {idxs} with C={C}."
            )
    # Build unions using index_select
    out_chans = []
    tgt_chans = []
    for p in parent_names:
        idxs = sorted(set(leaf_index[l] for l in desc[p]))
        if len(idxs) == 0:
            # If ever empty, make an explicit zero channel (prevents CUDA assert)
            mX = torch.zeros((B,1,H,W), dtype=dtypeX, device=dev)
            mY = torch.zeros((B,1,H,W), dtype=dtypeY, device=dev)
        else:
            idx_tensor = torch.as_tensor(idxs, dtype=torch.long, device=dev)
            selX = torch.index_select(X, dim=1, index=idx_tensor)
            selY = torch.index_select(Y, dim=1, index=idx_tensor)
            # union over the selected leaves (treat any >0 as positive)
            mX = (selX > 0).any(dim=1, keepdim=True).to(dtypeX)
            mY = (selY > 0).any(dim=1, keepdim=True).to(dtypeY)
        out_chans.append(mX)
        tgt_chans.append(mY)
    out_parents    = [torch.cat(out_chans, dim=1)]
    target_parents = [torch.cat(tgt_chans, dim=1)]
    return out_parents, target_parents, parent_names



# combines all per level classes into a single level
def combine_levels(
    leaves_list,
    parents_list,
    tree: dict,
    leaf_order=None,
    parent_order=None,
):
    # unwrap inputs
    X_leaves = leaves_list[0]
    X_par    = parents_list[0]
    B, _, H, W = X_leaves.shape
    device = X_leaves.device
    # derive structure
    levels = levels_bfs(tree)
    children = children_map(tree)
    # classify leaves/parents by tree
    all_names = [n for lvl in levels for n in lvl]
    is_leaf = {n: (len(children.get(n, [])) == 0) for n in all_names}
    leaf_names_tree   = [n for n in all_names if is_leaf[n]]
    parent_names_tree = [n for n in all_names if not is_leaf[n]]
    # default channel orders (BFS-derived) if not provided
    if leaf_order is None:
        # BFS leaves order from the tree itself
        leaf_order = leaf_names_tree
    if parent_order is None:
        # BFS parents order from the tree itself
        parent_order = parent_names_tree
    leaf_index   = {name: i for i, name in enumerate(leaf_order)}
    parent_index = {name: i for i, name in enumerate(parent_order)}
    # quick sanity check
    missing_leaves  = [n for n in leaf_names_tree   if n not in leaf_index]
    missing_parents = [n for n in parent_names_tree if n not in parent_index]
    if missing_leaves:
        raise KeyError(f"leaf_order is missing leaves: {missing_leaves}")
    if missing_parents:
        raise KeyError(f"parent_order is missing parents: {missing_parents}")
    # stitches each level
    out_per_level = []
    for lvl_names in levels:
        chans = []
        for n in lvl_names:
            if is_leaf[n]:
                idx = leaf_index[n]
                chans.append(X_leaves[:, idx:idx+1])
            else:
                idx = parent_index[n]
                chans.append(X_par[:, idx:idx+1])
        if len(chans) == 0:
            out_per_level.append(torch.zeros((B,0,H,W), device=device, dtype=X_leaves.dtype))
        else:
            out_per_level.append(torch.cat(chans, dim=1))
    return out_per_level




def build(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # if args.pre_split_val == True:
    img_path = os.path.join(args.root, "images", '*')
    print('Final Image Path: ', img_path)
    if not os.path.exists(os.path.dirname(img_path)):
        print('Image path does not exist (try adding /val to path if not using K fold val):', os.path.dirname(img_path))
        sys.exit()
    input_paths = sorted(glob.glob(img_path))
    lbl_path = os.path.join(args.root, "labels", '*') 
    print('Final label Path: ', lbl_path)
    if not os.path.exists(os.path.dirname(lbl_path)):
        print('Label path does not exist (try adding /val to path if not using K fold val):', os.path.dirname(lbl_path))
        sys.exit()
    target_paths = sorted(glob.glob(lbl_path))
        
    # loads tree json file
    class_tree = json.load(open(args.tree_root))
    # loads class map csv file as a dataframe
    class_map = pd.read_csv(args.class_map)

    if args.model_type == 0:
        args.num_classes = get_classes(class_tree, full=False, final_counts=[])
        args.num_classes_full = get_classes(class_tree, full=True, final_counts=[])
    else:
        args.num_classes = get_classes(class_tree, full=True, final_counts=[])
        args.num_classes_full = args.num_classes
    
    _, test_dataloader, _ = dataloaders.get_dataloaders(
        input_paths, target_paths, class_tree, class_map, batch_size=1, img_size=args.img_size, types='Predict', workers_num=args.num_workers, num_classes=sum(args.num_classes), model_type=args.model_type
    )
    
    # args.num_classes = get_classes(class_tree)



    accuracy = performance_metrics.Accuracy()
    iou = performance_metrics.Jaccardindex()
    perf = performance_metrics.DiceScore()
    precision = performance_metrics.Precision()
    recall = performance_metrics.Recall()

    # gets number of channels for input image
    num_channels = imread(input_paths[0]).shape[-1] if len(imread(input_paths[0]).shape) == 3 else 1
    # loads model 
    if args.model_select == 0:
        model = models.UNet(size=args.img_size, n_channels=num_channels, hierarchy=class_tree, model_type=args.model_type)
    elif args.model_select == 1:
        # implement new model here
        model = models.HighResolutionNet(config=config, hierarchy=class_tree, model_type=args.model_type)
        if args.model_weights != '':
            model.init_weights(args.model_weights, device)
    # model = models.UNet(size=args.img_size, n_channels=num_channels, hierarchy=class_tree, model_type=args.model_type)
    
    
    state_dict = torch.load(
        os.path.join(args.model_weights)
    )
    try:
        model.load_state_dict(state_dict["model_state_dict"])
    except:
        pass
    model.to(device)

    # get param numbers
    print('Number of model parameters: ', sum(p.numel() for p in model.parameters() if p.requires_grad))

    return device, test_dataloader, accuracy, iou, perf, precision, recall, model, target_paths, class_tree, class_map


@torch.no_grad()
def predict(args):

    temp_root = copy.deepcopy(args.root)
    temp_save = copy.deepcopy(args.save_loc)
    temp_weights = copy.deepcopy(args.model_weights)
    # loops folds
    if args.inc_cross_val == True:
        folds = args.folds
    else:
        folds = 1
    for fold in range(folds):
        fold_n = fold + 1
        args.root = temp_root[fold]
        args.model_weights = temp_weights[fold]

        if args.inc_cross_val == True:
            if not os.path.exists("./Predictions"):
                os.makedirs("./Predictions")
            if not os.path.exists("./Predictions/{}".format(args.save_loc)):
                os.makedirs("./Predictions/{}".format(args.save_loc))
            args.save_loc = os.path.join(temp_save, "fold_" + str(fold_n))
            
        if args.save_images:
            # if not os.path.exists(args.save_loc):
            #     args.save_loc = os.path.join(os.curdir, args.save_loc)
            
            if not os.path.exists("./Predictions"):
                os.makedirs("./Predictions")
            if not os.path.exists("./Predictions/{}".format(args.save_loc)):
                os.makedirs("./Predictions/{}".format(args.save_loc))

        print("Fold: ", fold_n)
        print("Root: ", args.root)

        device, test_dataloader, Accuracy, Iou, perf_measure, Precision, Recall, model, target_paths, class_tree, class_map = build(args)

        if args.save_images:
            for items in range(sum(args.num_classes_full)):
                if not os.path.exists("./Predictions/{}/{}".format(args.save_loc, items)):
                    os.makedirs("./Predictions/{}/{}".format(args.save_loc, items))



        t = time.time()
        model.eval()
        perf_accumulator = []

        IoU2 = []
        dice2 = []
        precision2 = []
        recall2 = []
        accuracy2 = []
        # superIOU, superACC, superPERF, superPREC, superRECA = [], [], [], [], []

        clssMetrics2 = []

        # sets up dictionary for class metrics
        for clss in range(sum(args.num_classes_full)):
            # creates a dictionary in classMetrics
            clssMetrics2.append({'accuracy': [], 'iou': [], 'dice': [], 'precision': [], 'recall': []})


        for i, (data, target) in enumerate(test_dataloader):
            data, target = data.to(device), target.to(device)# [i.to(device) for i in target]


            # spits target into sublists structure for each level of the hierarchy
            if args.model_type == 1:
                targets = []
                start_idx = 0
                for num_classes in args.num_classes:
                    end_idx = start_idx + num_classes
                    targets.append(target[:, start_idx:end_idx, :, :])
                    start_idx = end_idx
            else:
                targets = [target]

            # output = model(data)
            if args.model_select == 0:
                _, output_logits = model(data, type=args.model_type, hierarchy=class_tree) # model output is structured as [l1, l2, ...] of (batch, classes, h, w) where classes start at 0 at each level
            elif args.model_select == 1:
                _, output_logits = model(data)
                # for img_itr in range(len(output_class)):
                #     output_class[img_itr] = torch.where(output_class[img_itr] > 0.5, 1, 0)
            # output_class, output_logits = model(data, model_select=args.model_select, type=args.model_type, hierarchy=class_tree) # model output is structured as [l1, l2, ...] of (batch, classes, h, w) where classes start at 0 at each level

            if args.model_type == 0:

                # IoU, dice, precision, recall, accuracy, clssMetrics = [], [], [], [], [], []
                clssMetrics = []
                # sets up dictionary for class metrics
                for clss in range(sum(args.num_classes_full)):
                    # creates a dictionary in classMetrics
                    clssMetrics.append({'accuracy': [], 'iou': [], 'dice': [], 'precision': [], 'recall': []})

                # softmax
                output_class = F.softmax(output_logits, dim=1)
                # chooses class number
                output_class = torch.argmax(output_class, dim=1)
                # one hot encoding
                output_class = [F.one_hot(output_class, num_classes=sum(args.num_classes)).permute(0, 3, 1, 2).float()]
                output_logits = [output_logits]
                # bins output results to 0 or 255
                # output_class = np.where(output_class > 0.5, 255, 0)

                target = [target]
                # creates a tensor of zeros with the same shape as output_class and target
                # temp_out_class = [torch.zeros((output_class[0].shape[0], output_class[0].shape[1], output_class[0].shape[2], output_class[0].shape[3]), device=output_class[0].device)]
                # temp_target = [torch.zeros((target[0].shape[0], target[0].shape[1], target[0].shape[2], target[0].shape[3]), device=target[0].device)]
                # replcaes dim 1 class 0 with the correct values from output_class and target
                # temp_out_class[0][:, 0, :, :] = output_class[0][:, 0, :, :]
                # temp_target[0][:, 0, :, :] = target[0][:, 0, :, :]
                # temp_out_class = [output_class[0].clone()]
                # temp_target = [target[0].clone()]
                # gets parent classes
                # name_to_index = {row['class_name']: idx for idx, row in class_map.iterrows()}
                name_to_index = {name: idx for idx, name in enumerate([name for name in bfs_order(class_tree) if not children_map(class_tree).get(name)])}
                parent_class, parent_target, parent_class_numbers = get_parent_masks(output_class, target, class_tree, name_to_index)
                leaf_order = [i for i in name_to_index.keys()]
                parent_order = [n for n in bfs_order(class_tree) if children_map(class_tree).get(n)]
                output_class = combine_levels(output_class, parent_class, class_tree, leaf_order, parent_order)
                targets = combine_levels(target, parent_target, class_tree, leaf_order, parent_order)
                # traverses parent_class[0] and just shows parent_class image (converts 1 to 255)
                # for j in target:
                #     for i in j[0]:
                #         img = torch.where(i == 1.0, 255.0, i)
                #         plt.imshow(img.cpu().numpy())
                #         plt.show()

                # parent_class, parent_target,_, _, _, parent_class_numbers = get_parent_masks(output_class, target, class_tree, args.num_classes, temp_out_class, temp_target, 0, False, [])
                # replaces dim=1 position 0 with 1.0 if all other classes are 0.0
                # parent_class[0][:, 0, :, :] = torch.where(torch.sum(parent_class[0][:, 1:, :, :], dim=1) == 0.0, 1.0, parent_class[0][:, 0, :, :])
                # parent_target[0][:, 0, :, :] = torch.where(torch.sum(parent_target[0][:, 1:, :, :], dim=1) == 0.0, 1.0, parent_target[0][:, 0, :, :])
                
                # output_class, _, _ = update_classes(output_class, class_tree, args.num_pred_classes, args.num_classes, output_class_copy)
                # clssMetrics, _, _, _, _, _, _ = get_metrics(parent_class, parent_target, [], [], [], [], [], Accuracy, Iou, perf_measure, Precision, Recall, device, clssMetrics, args, child_trig=False)
                eval_targets = targets.copy()
                for targ_itr in range(len(targets)):
                    # turns output_class pixels to 0 if the same position in targets is -1
                    output_class[targ_itr] = torch.where(targets[targ_itr] == -1, 0, output_class[targ_itr])
                    eval_targets[targ_itr] = torch.where(targets[targ_itr] == -1, 0, targets[targ_itr])
                clssMetrics2, accuracy2, IoU2, dice2, precision2, recall2, pref_no_bg = get_metrics(output_class, eval_targets, accuracy2, IoU2, dice2, precision2, recall2, Accuracy, Iou, perf_measure, Precision, Recall, device, clssMetrics2, args) # , child_trig=True, parent_metrics = clssMetrics, parent_metric_posits = parent_class_numbers
                # clssMetrics2, accuracy2, IoU2, dice2, precision2, recall2, pref_no_bg = get_metrics(output_class, targets, accuracy2, IoU2, dice2, precision2, recall2, Accuracy, Iou, perf_measure, Precision, Recall, device, clssMetrics2, args)
            else:
                output_class_temp = []
                for out_itr in range(len(output_logits)):
                    output_class = F.softmax(output_logits[out_itr], dim=1)
                    # chooses class number
                    output_class = torch.argmax(output_class, dim=1)
                    # one hot encoding
                    output_class_temp.append(F.one_hot(output_class, num_classes=args.num_classes[out_itr]).permute(0, 3, 1, 2).float())
                output_class = output_class_temp
                # converts -1 in target to 0
                eval_targets = targets.copy()
                for targ_itr in range(len(targets)):
                    # turns output_class pixels to 0 if the same position in targets is -1
                    output_class[targ_itr] = torch.where(targets[targ_itr] == -1, 0, output_class[targ_itr])
                    eval_targets[targ_itr] = torch.where(targets[targ_itr] == -1, 0, targets[targ_itr])
                clssMetrics2, accuracy2, IoU2, dice2, precision2, recall2, pref_no_bg = get_metrics(output_class, eval_targets, accuracy2, IoU2, dice2, precision2, recall2, Accuracy, Iou, perf_measure, Precision, Recall, device, clssMetrics2, args)

            # updates clssMetrics2 with the metrics from clssMetrics 
            # tempAcc, tempIoU, tempDice, tempPrecision, tempRecall = [], [], [], [], []
            # if args.model_type == 0:
            #     for clss in parent_class_numbers:
            #         clssMetrics2.insert(clss, clssMetrics[clss])
            #         tempAcc.append(clssMetrics[clss]['accuracy'][0])
            #         tempIoU.append(clssMetrics[clss]['iou'][0])
            #         tempDice.append(clssMetrics[clss]['dice'][0])
            #         tempPrecision.append(clssMetrics[clss]['precision'][0])
            #         tempRecall.append(clssMetrics[clss]['recall'][0])
            #         pref_no_bg = torch.cat((pref_no_bg[:clss-1], torch.tensor([clssMetrics[clss]['dice'][0]], device=pref_no_bg.device), pref_no_bg[clss-1:]))
            #     accuracy2 = [sum(accuracy2)/len(accuracy2)]
            #     IoU2 = [sum(IoU2)/len(IoU2)]
            #     dice2 = [sum(dice2)/len(dice2)]
            #     precision2 = [sum(precision2)/len(precision2)]
            #     recall2 = [sum(recall2)/len(recall2)]



            # calculates metrics
            # tstClassMet, tstAcc, tstIoU, tstDice, tstPrecision, tstRecall, superIOU, superACC, superPERF, superPREC, superRECA, _ = get_metrics(output, target, accuracy2, IoU2, dice2, precision2, recall2, superIOU, superACC, superPERF, superPREC, superRECA, Accuracy, Iou, perf_measure, Precision, Recall, device, clssMetrics2, args, False, None, None, None)        

            # perf_accumulator.append(torch.mean(perf_measure(output, target, device, args.num_classes)[1][1:]).item())
            perf_accumulator.append(torch.mean(pref_no_bg).item())
            if i + 1 < len(test_dataloader):
                print(
                    "\rTest: [{}/{} ({:.1f}%)]\tAverage performance: {:.6f}\tTime: {:.6f}".format(
                        i + 1,
                        len(test_dataloader),
                        100.0 * (i + 1) / len(test_dataloader),
                        np.mean(perf_accumulator),
                        time.time() - t,
                    ),
                    end="",
                )
            else:
                print(
                    "\rTest: [{}/{} ({:.1f}%)]\tAverage performance: {:.6f}\tTime: {:.6f}".format(
                        i + 1,
                        len(test_dataloader),
                        100.0 * (i + 1) / len(test_dataloader),
                        np.mean(perf_accumulator),
                        time.time() - t,
                    )
                )

            # find probabilities of each class in output
            # output = F.softmax(output, dim = 1)
            # find the most likely class in output
            # output = torch.argmax(output, dim=1)
            # split output into a list of tensors, each tensor conains the pixels of a single class with 0 and 255 values
            # outputs = performance_metrics.split_targets(output, args.num_classes, pos_class_val=255)
            # loops through each class and saves
            # for j in range(len(outputs)):
            #     predicted_map = np.squeeze(np.array(outputs[j].cpu()))
            #     cv2.imwrite("./Predictions/{}/{}/{}".format(args.train_dataset, str(j), os.path.basename(target_paths[i])), predicted_map * 255)
            # if args.save_images_batch == True and inital == True:
            #     if epoch % args.save_images_batch_num == 0 or epoch == 1:
            #         print('Saving batch images...')
                    
            #         for lvl in range(len(output_class)):
            #             # takes the first image in the batch dim 0 and splits it by class dim 1
            #             first_image = output_class[lvl][0,:,:,:]
            #             # splits first_image by dim 0 into a list
            #             first_image = torch.split(first_image, 1, dim=0)
            #             for clss in range(len(first_image)):
            #                 save_clss = clss + sum(args.num_classes[:lvl])
            #                 predicted_map = np.squeeze(np.array(first_image[clss].cpu()))
            #                 cv2.imwrite(os.path.join(save_loc, "images", str(save_clss), "Epoch"+str(epoch)+".png"), (predicted_map * 255).astype(np.uint8))
            #         inital = False
            #TODO: traverse tree and combine child classes into parent class masks and insert them into the correct location in the tensor

            # if args.model_type == 0:
            #     # traverses parent_class_numbers and inserts parent_class value at position parent_class_numbers
            #     for clss in parent_class_numbers:
            #         output_class[0] = torch.cat((output_class[0][:, :clss], parent_class[0][:, clss].unsqueeze(1), output_class[0][:, clss:]), dim=1)
            if args.save_images == True:
                save_clss = 0
                for lvl in range(len(output_class)):
                    # takes the first image in the batch dim 0 and splits it by class dim 1
                    first_image = output_class[lvl][0,:,:,:]
                    # splits first_image by dim 0 into a list
                    first_image = torch.split(first_image, 1, dim=0)
                    for clss in range(len(first_image)):
                        # for img_itr in range(len(output_class)):
                        img_out = torch.where(first_image[clss] > 0.5, 1, 0)
                        predicted_map = np.squeeze(np.array(img_out.cpu()))
                        cv2.imwrite(os.path.join("./Predictions/{}/{}".format(args.save_loc, save_clss), os.path.basename(target_paths[i])), (predicted_map * 255).astype(np.uint8))
                        save_clss += 1


        # for met in range(len(clssMetrics2)):
        #     for key in clssMetrics2[met]:
        #         clssMetrics2[met][key] = np.mean(clssMetrics2[met][key])
        

        print("FINISHED TESTING")


        # print val metrics
        print('Validation Accuracy: ', np.mean(accuracy2).item())
        print('Validation IoU: ', np.mean(IoU2).item())
        print('Validation Dice: ', np.mean(dice2).item())
        print('Validation Precision: ', np.mean(precision2).item())
        print('Validation Recall: ', np.mean(recall2).item())




        # prints val metrics for each class
        for clss in range(len(clssMetrics2)):
            print('Class: ', clss)
            print('Validation Accuracy: ', np.mean(np.array(clssMetrics2[clss]['accuracy'])).item())
            print('Validation IoU: ', np.mean(np.array(clssMetrics2[clss]['iou'])).item())
            print('Validation Dice: ', np.mean(np.array(clssMetrics2[clss]['dice'])).item())
            print('Validation Precision: ', np.mean(np.array(clssMetrics2[clss]['precision'])).item())
            print('Validation Recall: ', np.mean(np.array(clssMetrics2[clss]['recall'])).item())
        
        # if args.calc_super != False:
        #     print('Superclass Metrics class 1 and 2')
        #     print('Superclass Val Accuracy: ', np.mean(superACC).item())
        #     print('Superclass Val IoU: ', np.mean(superIOU).item())
        #     print('Superclass Val Dice: ', np.mean(superPERF).item())
        #     print('Superclass Val Precision: ', np.mean(superPREC).item())
        #     print('Superclass Val Recall: ', np.mean(superRECA).item())



        # save metrics to txt file in save location
        # Save overall metrics
        with open("./Predictions/{}/metrics.csv".format(args.save_loc), "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Type", "Class", "Accuracy", "IoU", "Dice", "Precision", "Recall"])
            writer.writerow(["Average", "All", 
                            np.mean(accuracy2).item(), 
                            np.mean(IoU2).item(), 
                            np.mean(dice2).item(), 
                            np.mean(precision2).item(), 
                            np.mean(recall2).item()])
            
            # Save class-specific metrics
            for clss in range(len(clssMetrics2)):
                writer.writerow(["Class", clss, 
                                np.mean(np.array(clssMetrics2[clss]['accuracy'])).item(), 
                                np.mean(np.array(clssMetrics2[clss]['iou'])).item(), 
                                np.mean(np.array(clssMetrics2[clss]['dice'])).item(), 
                                np.mean(np.array(clssMetrics2[clss]['precision'])).item(), 
                                np.mean(np.array(clssMetrics2[clss]['recall'])).item()])
        
            


def get_args():
    parser = argparse.ArgumentParser(
        description="Make predictions on specified dataset"
    )
    parser.add_argument("--save-loc", type=str, required=True) # the save name for the predictions
    parser.add_argument("--full-ds", type=str, default="False", dest="root")
    parser.add_argument("--tree-root", type=str, required=True)
    parser.add_argument("--class-map", type=str, required=True)
    # parser.add_argument("--pre-split-val", type=str, required=True, default="False")
    parser.add_argument("--model-weights", type=str, required=True)
    # parser.add_argument("--pretrain-weights", type=str, required=True)
    parser.add_argument("--img-size", type=int, default=352)
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=-1)
    parser.add_argument("--save-images", type=str, default="True")
    # parser.add_argument("--calc-super", type=str, default="False")
    parser.add_argument("--include-background", type=str, default="False")
    parser.add_argument("--include-std-div", type=str, default="False")
    parser.add_argument("--model-type", type=int, default=0, choices=[0, 1], help="0 for normal, 1 for hierarchical")
    parser.add_argument("--model-select", type=int, default=0, choices=[0, 1], help="0 for UNet, 1 for")
    parser.add_argument("--inc-cross-val", type=str, default="True") 
    parser.add_argument("--folds", type=int, default=0)
    parser.add_argument("--config", type=str, default='')

    return parser.parse_args()


def main():
    args = get_args()

    if args.model_select == 1:
        update_config(config, args)

    args.inc_cross_val = True if args.inc_cross_val == "True" else False

    # if there is cross validation, set reads data-root and val-dataset as a list of paths and save-path as an initial file name with the fold number added
    # if args.inc_cross_val:
    #     args.root = args.root
    #     args.val_dataset = args.val_dataset

    # check if save path, tree root, class map and model weights exist
    # if not os.path.exists(args.save_path):
    #     args.save_path = os.path.join(os.curdir, args.save_path)
        # if not os.path.exists(os.path.join(args.save_path.split('/')[:-1])):
        #     print("Save path does not exist")
        #     sys.exit()



    # if not os.path.exists(args.tree_root):
    #     args.tree_root = os.path.join(os.curdir, args.tree_root)
    #     if not os.path.exists(args.tree_root):
    #         print("Tree root does not exist")
    #         sys.exit()
    # if not os.path.exists(args.class_map):
    #     args.class_map = os.path.join(os.curdir, args.class_map)
    #     if not os.path.exists(args.class_map):
    #         print("Class map does not exist")
    #         sys.exit()
    
    # if args.inc_cross_val:
    #     if not os.path.exists(args.root):
    #         args.root = os.path.join(os.curdir, args.root)
    #         if not os.path.exists(args.root):
    #             print("Root dataset does not exist")
    #             sys.exit()
    #     if not os.path.exists(args.val_dataset):
    #         args.val_dataset = os.path.join(os.curdir, args.val_dataset)
    #         if not os.path.exists(args.val_dataset):
    #             print("Validation dataset does not exist")
    #     train_locs = []
    #     val_locs = []
    #     for i in range(args.folds):
    #         train_locs.append(os.path.join(args.root, "fold_" + str(i+1), "train"))
    #         val_locs.append(os.path.join(args.root, "fold_" + str(i+1), "val"))
    #         # if not os.path.exists(train_locs[i]):
    #         #     os.makedirs(train_locs[i])
    #         # if not os.path.exists(val_locs[i]):
    #         #     os.makedirs(val_locs[i])
    #     args.root = train_locs
    #     args.val_dataset = val_locs
    # else:
    #     if args.root != 'None' and not os.path.exists(args.val_dataset):
    #         if args.val_dataset == 'None':
    #             args.val_dataset = os.path.join(args.root, "val")
    #         else:
    #             args.val_dataset = os.path.join(os.curdir, args.val_dataset, "val")
    #         if not os.path.exists(args.val_dataset):
    #             print("Validation dataset does not exist")
    #     if args.root != 'None' and not os.path.exists(args.root):
    #         args.root = os.path.join(os.curdir, args.root)
    #         if not os.path.exists(args.root):
    #             print("Root dataset does not exist")
    #             sys.exit()
    #     args.root = [os.path.join(args.root, "train")]
    #     args.val_dataset = [args.val_dataset]


    if args.inc_cross_val:
        if not os.path.exists(args.root):
            args.root = os.path.join(os.curdir, args.root)
            if not os.path.exists(args.root):
                print("Root dataset does not exist")
                sys.exit()
        if not os.path.exists(args.model_weights):
            args.model_weights = os.path.join(os.curdir, args.model_weights)
            if not os.path.exists(args.model_weights):
                print("Root dataset does not exist")
                sys.exit()
        # if not os.path.exists(args.val_dataset):
        #     args.val_dataset = os.path.join(os.curdir, args.val_dataset)
        #     if not os.path.exists(args.val_dataset):
        #         print("Validation dataset does not exist")
        #         sys.exit()
        val_locs = []
        model_locs = []
        # val_locs = []
        for i in range(args.folds):
            if os.path.exists(os.path.join(args.root, "fold_" + str(i+1), "val")):
                val_locs.append(os.path.join(args.root, "fold_" + str(i+1), "val"))
                model_locs.append(os.path.join(args.model_weights, "fold_" + str(i+1), "best.pt"))
            else:
                val_locs.append(args.root)
                model_locs.append(os.path.join(args.model_weights, "fold_" + str(i+1), "best.pt"))
            # val_locs.append(os.path.join(args.val_dataset, "fold_" + str(i+1), "val"))
            # if not os.path.exists(train_locs[i]):
            #     os.makedirs(train_locs[i])
            # if not os.path.exists(val_locs[i]):
            #     os.makedirs(val_locs[i])
        args.root = val_locs
        args.model_weights = model_locs
        # args.val_dataset = val_locs
    else:
        # if args.val_dataset != 'None' and not os.path.exists(args.val_dataset):
        #     args.val_dataset = os.path.join(os.curdir, args.val_dataset)
        #     if not os.path.exists(args.val_dataset):
        #         print("Validation dataset does not exist")
        #         sys.exit()
        # if not os.path.exists(args.val_dataset):
        #     args.val_dataset = os.path.join(os.curdir, args.val_dataset)
        #     if not os.path.exists(args.val_dataset):
        #         print("Validation dataset does not exist")
        #         sys.exit()
        if args.root != 'None' and not os.path.exists(args.root):
            args.root = os.path.join(os.curdir, args.root)
            if not os.path.exists(args.root):
                print(args.root)
                print("Root dataset does not exist. try adding /val to your path if you are not using 5 fold validation")
                sys.exit()
        args.root = [args.root]
        args.model_weights = [args.model_weights]
        # args.val_dataset = [args.val_dataset]
        

    # converts string to boolean for some args 
    # args.full_ds = True if args.full_ds == "True" else False
    # args.pre_split_val = True if args.pre_split_val == "True" else False
    args.save_images = True if args.save_images == "True" else False
    # args.calc_super = True if args.calc_super == "True" else False
    args.include_background = True if args.include_background == "True" else False
    args.include_std_div = True if args.include_std_div == "True" else False


    predict(args)


if __name__ == "__main__":
    main()

