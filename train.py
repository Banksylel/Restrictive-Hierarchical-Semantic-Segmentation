import sys
import os
import argparse
import time
import numpy as np
import glob
import csv
import copy
from tree_util import create_tree_from_textfile, add_channels, add_levels, find_depth, getTreeList, update_channels

import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from skimage.io import imread
from skimage.transform import resize

from Data import dataloaders
from Models import models
from Metrics import performance_metrics
from Metrics import losses
import math
import json

from config import config
from config import update_config

# WARNING: THIS SUPPRESSES WARNINGS, REMOVE IF YOU WANT TO SEE WARNINGS
import warnings
import pandas as pd
warnings.filterwarnings("ignore")



def get_metrics(output, target, accuracy, IoU, dice, precision, recall, Accuracy, Iou, perf_measure, Precision, Recall, device, clssMetrics, args, child_trig=True, parent_metrics=[], parent_metric_posits=[]):
       
        newIOU, newACC, newPERF, newPREC, newRECA = torch.empty(0, dtype=torch.float32, device=device), torch.empty(0, dtype=torch.float32, device=device), torch.empty(0, dtype=torch.float32, device=device), torch.empty(0, dtype=torch.float32, device=device), torch.empty(0, dtype=torch.float32, device=device)
        for outs in range(len(output)):
            if outs == 0:
                child = False
            else:
                child = True
            clss_num = target[outs].shape[1]
            newIOU = torch.cat((newIOU, Iou(output[outs], target[outs], device, clss_num, child)))
            newACC = torch.cat((newACC, Accuracy(output[outs], target[outs], device, clss_num, child)))
            newPERF = torch.cat((newPERF, perf_measure(output[outs], target[outs], device, clss_num, child)))
            newPREC = torch.cat((newPREC, Precision(output[outs], target[outs], device, clss_num, child)))
            newRECA = torch.cat((newRECA, Recall(output[outs], target[outs], device, clss_num, child)))

        perf_no_bg = newPERF[1:]

        if parent_metrics != []:
            for posit in parent_metric_posits:
                # adds the metrics from parent_metrics at positions parent_metric_posits into the current metrics
                newACC = torch.cat((newACC[:posit], torch.tensor([parent_metrics[posit]["accuracy"][-1]], device=device), newACC[posit:]))
                newIOU = torch.cat((newIOU[:posit], torch.tensor([parent_metrics[posit]["iou"][-1]], device=device), newIOU[posit:]))
                newPERF = torch.cat((newPERF[:posit], torch.tensor([parent_metrics[posit]["dice"][-1]], device=device), newPERF[posit:]))
                newPREC = torch.cat((newPREC[:posit], torch.tensor([parent_metrics[posit]["precision"][-1]], device=device), newPREC[posit:]))
                newRECA = torch.cat((newRECA[:posit], torch.tensor([parent_metrics[posit]["recall"][-1]], device=device), newRECA[posit:]))
                perf_no_bg = torch.cat((perf_no_bg[:posit-1], torch.tensor([parent_metrics[posit]["dice"][-1]], device=device), perf_no_bg[posit-1:]))

        # updates the total metrics for all classes
        accuracy.append(torch.mean(newACC).item())
        IoU.append(torch.mean(newIOU).item())
        dice.append(torch.mean(newPERF).item())
        precision.append(torch.mean(newPREC).item())
        recall.append(torch.mean(newRECA).item())

        for clss in range(len(newACC)):
            # add calculates the per class metrics 
            clssMetrics[clss]['accuracy'].append(newACC[clss].item())
            clssMetrics[clss]['iou'].append(newIOU[clss].item())
            clssMetrics[clss]['dice'].append(newPERF[clss].item())
            clssMetrics[clss]['precision'].append(newPREC[clss].item())
            clssMetrics[clss]['recall'].append(newRECA[clss].item())


        return(clssMetrics, accuracy, IoU, dice, precision, recall, perf_no_bg)



# traverses class tree to find the number of classes in each level of the hierarchy
def get_classes(class_tree, full=False, final_counts=[]):

    counts: list[int] = []

    def traverse_tree(node: dict, depth: int):
        # Ensure we have a slot for this depth
        if len(counts) <= depth:
            counts.append(0)

        for _, child in node.items():
            is_leaf = not isinstance(child, dict) or (isinstance(child, dict) and len(child) == 0)

            if full or is_leaf:
                counts[depth] += 1

            # Recurse into non-empty dict children
            if isinstance(child, dict) and child:
                traverse_tree(child, depth + 1)

    traverse_tree(class_tree, 0)
    return counts



# calculates per level loss and hierarchical consistency loss
def get_loss(output_logits,
             targets,
             lossFuncts,
             levelLoss,
             level_weights=None,
             loss=0.0,
             lvlLossGrad=[],
             cur_level=None,
             cur_epoch=None,
             pretrain_epoch=None,
             probs_per_level=None,
             model=None):

    # sets depths
    if pretrain_epoch is not None:
        cur_level_cap = int(min(len(output_logits)-1, (cur_epoch // pretrain_epoch)))
    total_levels = len(output_logits)
    if len(levelLoss) != total_levels:
        levelLoss[:] = [0.0] * total_levels

    # supervised per-level losses
    for L in range(total_levels):
        if pretrain_epoch is not None and L > cur_level_cap:
            continue
        level_weight = None if level_weights is None else level_weights[L]
        loss_ce   = lossFuncts[L][0](output_logits[L], targets[L], class_weight=level_weight, logits_input=True)
        loss_dice = lossFuncts[L][1](output_logits[L], targets[L], class_weight=level_weight, logits_input=True)
        if loss_ce is not None:
            loss += loss_ce
            levelLoss[L] = (levelLoss[L] + loss_ce.item())
        if loss_dice is not None:
            loss += loss_dice
            levelLoss[L] = (levelLoss[L] + loss_dice.item())

    # hierarchical consistency
    if (probs_per_level is not None) and (model is not None) and hasattr(model, "levels") and hasattr(model, "parent_of"):
        loss += losses.hierarchical_consistency_loss(
            probs_per_level, model.levels, model.parent_of, reduction='mean'
        )


    return loss, lvlLossGrad, levelLoss







# training loop
def train_epoch(model, device, train_loader, optimizer, epoch, lossFuncts, args, class_tree, class_map, Accuracy, Iou, perf_measure, Precision, Recall, epoch_num):
    first_test = True
    loss, accuracy, IoU, dice, precision, recall = [], [], [], [], [], []
    levelLoss = []
    

    clssMetrics = []

    # sets up dictionary for class metrics
    for clss in range(sum(args.num_classes)):
        # creates a dictionary in classMetrics
        clssMetrics.append({'accuracy': [], 'iou': [], 'dice': [], 'precision': [], 'recall': []})  # creates a dictionary for each class



    t = time.time()
    model.train()
    loss_accumulator = []
    for batch_idx, (data, target) in enumerate(train_loader):
    
        data, target = data.to(device), target.to(device)


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



        # traverses class 
        optimizer.zero_grad()
 
        # model outputs class and logits per hierarchy. output_class cannot be backpropagated, output_logits can be backpropagated
        if args.model_select == 0:
            _, output_logits = model(data, type=args.model_type, hierarchy=class_tree)
        else:
            _, output_logits = model(data)

        if args.model_type == 0:
            # softmax
            output_class = F.softmax(output_logits, dim=1)
            # chooses class number
            output_class = torch.argmax(output_class, dim=1)
            # one hot encoding
            output_class = [F.one_hot(output_class, num_classes=sum(args.num_classes)).permute(0, 3, 1, 2).float()]
            output_logits = [output_logits]
            # changes -1.0 to 0.0
            # targets = [torch.where(targets[0] == -1.0, 0.0, targets[0])]
        else:
            output_class_temp = []
            for out_itr in range(len(output_logits)):
                output_class = F.softmax(output_logits[out_itr], dim=1)
                # chooses class number
                output_class = torch.argmax(output_class, dim=1)
                # one hot encoding
                output_class_temp.append(F.one_hot(output_class, num_classes=args.num_classes[out_itr]).permute(0, 3, 1, 2).float()) # sum(args.num_classes)
            output_class = output_class_temp


        eval_targets = targets.copy()
        for targ_itr in range(len(targets)):
            # turns output_class pixels to 0 if the same position in targets is -1
            output_class[targ_itr] = torch.where(targets[targ_itr] == -1, 0, output_class[targ_itr])
            eval_targets[targ_itr] = torch.where(targets[targ_itr] == -1, 0, targets[targ_itr])
        clssMetrics, accuracy, IoU, dice, precision, recall, _ = get_metrics(output_class, eval_targets, accuracy, IoU, dice, precision, recall, Accuracy, Iou, perf_measure, Precision, Recall, device, clssMetrics, args)
        del eval_targets


        # Choose probabilities for hierarchy terms only when hierarchical
        probs_per_level = output_class if args.model_type == 1 else None

        loss, lvlLossGrad, levelLoss = get_loss(output_logits, targets, lossFuncts, levelLoss, args.level_weights, 0.0, [], cur_epoch=epoch_num, pretrain_epoch=args.level0_pretrain_epochs, probs_per_level=probs_per_level, model=model, lambda_cons=1.0, lambda_kl=0.1)

        loss.backward()
        optimizer.step()





        loss_accumulator.append(loss.item())
        if batch_idx + 1 < len(train_loader):
            print(
                "\rTrain Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}\tTime: {:.6f}".format(
                    epoch,
                    (batch_idx + 1) * len(data),
                    len(train_loader.dataset),
                    100.0 * (batch_idx + 1) / len(train_loader),
                    loss.item(),
                    time.time() - t,
                ),
                end="",
            )
        else:
            print(
                "\rTrain Epoch: {} [{}/{} ({:.1f}%)]\tAverage loss: {:.6f}\tTime: {:.6f}".format(
                    epoch,
                    (batch_idx + 1) * len(data),
                    len(train_loader.dataset),
                    100.0 * (batch_idx + 1) / len(train_loader),
                    np.mean(loss_accumulator),
                    time.time() - t,
                )
            )
        
    # means the contents of each key in each dictionary in clssMetrics
    for met in range(len(clssMetrics)):
        for key in clssMetrics[met]:
            clssMetrics[met][key] = np.mean(clssMetrics[met][key])
        

    return np.mean(loss_accumulator).item(), clssMetrics, np.mean(accuracy).item(), np.mean(IoU).item(), np.mean(dice).item(), np.mean(precision).item(), np.mean(recall).item(), [i/(len(train_loader)*args.batch_size) for i in levelLoss]


@torch.no_grad()
def test(model, device, test_loader, epoch, Accuracy, Iou, perf_measure, Precision, Recall, args, save_loc, lossFuncts, class_tree, class_map):
    print('TESTING')
    IoU2 = []
    dice2 = []
    precision2 = []
    precision2 = []
    recall2 = []
    accuracy2 = []
    # superIOU, superACC, superPERF, superPREC, superRECA = [], [], [], [], []

    clssMetrics2 = []

    inital = True

    # sets up dictionary for class metrics
    for clss in range(sum(args.num_classes)):
        # creates a dictionary in classMetrics
        clssMetrics2.append({'accuracy': [], 'iou': [], 'dice': [], 'precision': [], 'recall': []})  # creates a dictionary for each class

    t = time.time()
    model.eval()
    perf_accumulator = []
    levelLossTest = []
    for batch_idx, (data, target) in enumerate(test_loader):

        data, target = data.to(device), target.to(device)

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
        


        if args.model_select == 0:
            output_class, output_logits = model(data, type=args.model_type, hierarchy=class_tree) # model output is structured as [l1, l2, ...] of (batch, classes, h, w) where classes start at 0 at each level
        elif args.model_select == 1:
            output_class, output_logits = model(data)
    
        if args.model_type == 0:
            # softmax
            output_class = F.softmax(output_logits, dim=1)
            # chooses class number
            output_class = torch.argmax(output_class, dim=1)
            # one hot encoding
            output_class = [F.one_hot(output_class, num_classes=sum(args.num_classes)).permute(0, 3, 1, 2).float()]
            output_logits = [output_logits]
        # metrics
        clssMetrics2, accuracy2, IoU2, dice2, precision2, recall2, pref_no_bg = get_metrics(output_class, targets, accuracy2, IoU2, dice2, precision2, recall2, Accuracy, Iou, perf_measure, Precision, Recall, device, clssMetrics2, args)

        probs_per_level = output_class if args.model_type == 1 else None
        loss, lvlLossGrad, levelLossTest = get_loss(output_logits, targets, lossFuncts, levelLossTest, args.level_weights, 0.0, [], probs_per_level=probs_per_level, model=model, lambda_cons=1.0, lambda_kl=0.1)
        lossTest = loss.item()


        # calculates perfect measure (dice) for all classes except background
        perf_accumulator.append(torch.mean(pref_no_bg).item())
        if batch_idx + 1 < len(test_loader):
            print(
                "\rTest  Epoch: {} [{}/{} ({:.1f}%)]\tAverage performance: {:.6f}\tTime: {:.6f}".format(
                    epoch,
                    batch_idx + 1,
                    len(test_loader),
                    100.0 * (batch_idx + 1) / len(test_loader),
                    np.mean(perf_accumulator),
                    time.time() - t,
                ),
                end="",
            )
        else:
            print(
                "\rTest  Epoch: {} [{}/{} ({:.1f}%)]\tAverage performance: {:.6f}\tTime: {:.6f}".format(
                    epoch,
                    batch_idx + 1,
                    len(test_loader),
                    100.0 * (batch_idx + 1) / len(test_loader),
                    np.mean(perf_accumulator),
                    time.time() - t,
                )
            )

        if args.save_images_batch == True and inital == True:
            if epoch % args.save_images_batch_num == 0 or epoch == 1:
                print('Saving batch images...')
                save_clss = 0
                for lvl in range(len(output_class)):
                    # takes the first image in the batch dim 0 and splits it by class dim 1
                    first_image = output_class[lvl][0,:,:,:]
                    # splits first_image by dim 0 into a list
                    first_image = torch.split(first_image, 1, dim=0)
                    for clss in range(len(first_image)):
                        predicted_map = np.squeeze(np.array(first_image[clss].cpu()))
                        # converts pixels values to 0 and 255 if below or above 0.5
                        predicted_map = np.where(predicted_map > 0.5, 1, 0)
                        cv2.imwrite(os.path.join(save_loc, "images", str(save_clss), "Epoch"+str(epoch)+".png"), (predicted_map * 255).astype(np.uint8))
                        save_clss += 1
                inital = False

    # means the contents of each key in each dictionary in clssMetrics
    for met in range(len(clssMetrics2)):
        for key in clssMetrics2[met]:
            clssMetrics2[met][key] = np.mean(clssMetrics2[met][key])

    print("FINISHED TESTING")
    return np.mean(perf_accumulator).item(), np.std(perf_accumulator).item(), clssMetrics2, np.mean(accuracy2).item(), np.mean(IoU2).item(), np.mean(dice2).item(), np.mean(precision2).item(), np.mean(recall2).item(), [i/(len(test_loader)*args.batch_size) for i in levelLossTest], lossTest

def build(args):

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # loads tree json file
    class_tree = json.load(open(args.tree_root))
    # loads class map csv file as a dataframe
    class_map = pd.read_csv(args.class_map)
    # class_map = {int(rows[0]): rows[1] for rows in class_map}

    # gets the number of classes in the class tree
    # if args.model_type == 0:
    #     args.num_classes = [sum(get_classes(class_tree, full=False))]
    # else:
    
    if args.model_type == 0:
        args.num_classes = get_classes(class_tree, full=False, final_counts=[])
    else:
        args.num_classes = get_classes(class_tree, full=True, final_counts=[])

    img_path = os.path.join(args.root, "images", '*')
    input_paths = sorted(glob.glob(img_path))
    depth_path = os.path.join(args.root, "labels", '*') 
    target_paths = sorted(glob.glob(depth_path))
    
    # Additional code to use a pre split val dataset
    if args.val_dataset != 'None':
        img_path2 = os.path.join(args.val_dataset, "images", '*')
        depth_path2 = os.path.join(args.val_dataset, "labels", '*')
        val_img_path = sorted(glob.glob(img_path2))
        val_target_path = sorted(glob.glob(depth_path2))
    else:
        val_img_path = 'None'
        val_target_path = 'None'
    
    if not val_img_path:
        print('Val Set Is Empty')
        sys.exit()

    
    train_dataloader, _, val_dataloader = dataloaders.get_dataloaders(
        input_paths, target_paths, class_tree, class_map, batch_size=args.batch_size, val_batch_size=args.val_batch, val_img=val_img_path, val_target=val_target_path, test_img='None', test_target='None', img_size=args.img_size, test_remove=args.test_remove, workers_num=args.num_workers, num_classes=sum(args.num_classes), model_type=args.model_type # , class_tree, class_map
    )
    
    loss = []
    for level in args.num_classes:
        loss.append([losses.CrossEntropyLoss(), losses.SoftDiceLoss(num_classes=level)])

    Accuracy = performance_metrics.Accuracy()
    Iou = performance_metrics.Jaccardindex()
    perf = performance_metrics.DiceScore()
    Precision = performance_metrics.Precision()
    Recall = performance_metrics.Recall()
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



    # change to unet
    # model = models.FCBFormer(size=args.img_size, weightPth=args.weight_path, n_classes=args.num_classes)

    # loads model weights from .pt file
    if args.model_weights != 'None' and args.model_select != 0:
        load_state_dict = torch.load(args.model_weights)#["model_state_dict"]
        curr_state_dict = model.state_dict()

        weights_loaded = []
        load_state = []
        curr_state = []
        # splits load and curr state into their components (in, up, down, out)
        for split_type in ["in", "up", "down", "out"]:
            load_state.append({k: v for k, v in load_state_dict.items() if split_type in k.split('.')[0]})
            curr_state.append({k: v for k, v in curr_state_dict.items() if split_type in k.split('.')[0]})

        # loads weights by name first if the same shape, then by key identifying numbers with the same shape.
        for types in range(len(curr_state)):
            for i, (k, v) in enumerate(curr_state[types].items()):
                break_trig = True
                for j, (k2, v2) in enumerate(load_state[types].items()):
                    # loads the same name sake
                    if k == k2 and v2.size() == v.size():
                        curr_state_dict[k] = load_state_dict[k2]
                        weights_loaded.append({k: k2})
                        break_trig = False
                        break
                    # loads by identifiers and number
                    elif k.split(".")[0] == k2.split(".")[0]: 
                        if k.split(".")[-1] == k2.split(".")[-1] and k.split(".")[-2] == k2.split(".")[-2] and v2.size() == v.size():
                            curr_state_dict[k] = load_state_dict[k2]
                            weights_loaded.append({k: k2})
                            break_trig = False
                            break

                if break_trig:
                    # randomises the current weights
                    curr_state_dict[k] = (2 * torch.rand(v.size()).to(device)) - 1

        model.load_state_dict(curr_state_dict)



    if args.mgpu == "true":
        model = nn.DataParallel(model)
    model.to(device)
    # if args.model_select == 0:
    if args.model_type == 0:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)


    return (
        device,
        train_dataloader,
        val_dataloader,
        loss,
        Accuracy,
        Iou,
        perf,
        Precision,
        Recall,
        model,
        optimizer,
        class_tree, 
        class_map
    )


def train(args):
    # deep copy of args
    temp_save_path = copy.deepcopy(args.save_path)
    temp_root = copy.deepcopy(args.root)
    temp_val_dataset = copy.deepcopy(args.val_dataset)
    # loops folds
    if args.inc_cross_val == True:
        folds = args.folds
    else:
        folds = 1
    for fold in range(folds):
        fold_n = fold + 1
        if fold_n != 1:
            continue
        args.save_path = os.path.join(temp_save_path, "fold_" + str(fold_n))
        args.root = temp_root[fold]
        args.val_dataset = temp_val_dataset[fold]
        print("Fold: ", fold_n)
        print("Root: ", args.root)
        print("Val Dataset: ", args.val_dataset)
        (
            device,
            train_dataloader,
            val_dataloader,
            lossFunc,
            Accuracy,
            Iou,
            perf,
            Precision,
            Recall,
            model,
            optimizer,
            class_tree, 
            class_map
        ) = build(args)

        save_loc = args.save_path
        fold_flag=False
        if os.path.exists(save_loc):
            # creates another folder of the same name + 1
            print("Save location already exists")
            if fold_flag == False:
                runNumtemp = os.path.split(save_loc)[-1].split('_')
                folName = runNumtemp[0]
                runNum = 1
                while os.path.exists(save_loc+'_'+str(runNum)):
                    runNum += 1
            save_loc = os.path.join(os.path.split(save_loc)[0], folName+ "_" + str(runNum), "fold_" + str(fold_n))
            fold_flag = True
        try:
            # Creates save folder
            os.makedirs(save_loc)
        except:
            pass
        print("Save Location: ", save_loc)
        if not os.path.exists(os.path.join(save_loc, "images")):
            os.makedirs(os.path.join(save_loc, "images"))
        
        for clss2 in range(sum(args.num_classes)):
            if not os.path.exists(os.path.join(save_loc, "images", str(clss2))):
                os.makedirs(os.path.join(save_loc, "images", str(clss2)))

        # if there is a .csv file in the save location, delete it
        if os.path.exists(os.path.join(save_loc, "metrics.csv")):
            os.remove(os.path.join(save_loc, "metrics.csv"))

        prev_best_test = None
        if args.lrs == "true":
            if args.lrs_min > 0:
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="max", factor=0.5, patience=3,  min_lr=args.lrs_min, verbose=True
                )
            else:
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="max", factor=0.5, patience=3, verbose=True
                )

        
        # checks if the number of epochs allow all levels to be trained to some extent
        # if (args.epochs/args.epochs_per_level <= len(args.num_classes) - 1):
        #     raise ValueError("Number of epochs per level is too low for the number of levels in the class tree. Child classes will not be trained.")
        # START TRAINING EPOCHS
        for epoch in range(1, args.epochs + 1):
            
            # checks epochs per level
            # if epoch % args.epochs_per_level == 0:
            #     max_level += 1
            # # checks max level if less than the number of total levels
            # if max_level >= len(args.num_classes):
            #     max_level = len(args.num_classes) - 1
            # try:
            loss, trnClassMet, trnAcc, trnIoU, trnDice, trnPrecision, trnRecall, levelLoss = train_epoch(
                model, device, train_dataloader, optimizer, epoch, lossFunc, args,class_tree, class_map, Accuracy, Iou, perf, Precision, Recall, epoch_num=epoch
            )
            test_measure_mean, test_measure_std, tstClassMet, tstAcc, tstIoU, tstDice, tstPrecision, tstRecall, levelLossTest, lossTest = test(
                model, device, val_dataloader, epoch, Accuracy, Iou, perf, Precision, Recall, args, save_loc, lossFunc, class_tree, class_map
            )
            # save metrics in train save location
            if not os.path.exists(os.path.join(save_loc, "metrics.csv")):
                with open(os.path.join(save_loc, "metrics.csv"), 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["Epoch", "Train Loss", "Train Level Loss", "Train Accuracy", "Train IoU", "Train Dice", "Train Precision", "Train Recall", "Train Class Metrics", "Val Loss", "Val Level Loss", "Val Accuracy", "Val IoU", "Val Dice", "Val Precision", "Val Recall", "Val Test Measure Mean", "Val Test Measure Std", "Val Class Metrics"])
                    writer.writerow([epoch, loss, levelLoss, trnAcc, trnIoU, trnDice, trnPrecision, trnRecall, trnClassMet, lossTest, levelLossTest, tstAcc, tstIoU, tstDice, tstPrecision, tstRecall, test_measure_mean, test_measure_std, tstClassMet])
            else:
                with open(os.path.join(save_loc, "metrics.csv"), 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, loss, levelLoss, trnAcc, trnIoU, trnDice, trnPrecision, trnRecall, trnClassMet, lossTest, levelLossTest, tstAcc, tstIoU, tstDice, tstPrecision, tstRecall, test_measure_mean, test_measure_std, tstClassMet])


            # print val metrics
            print('Validation Accuracy: ', tstAcc)
            print('Validation IoU: ', tstIoU)
            print('Validation Dice: ', tstDice)
            print('Validation Precision: ', tstPrecision)
            print('Validation Recall: ', tstRecall)

            # prints val metrics for each class
            for clss in range(len(tstClassMet)):
                print('Class: ', clss)
                print('Validation Accuracy: ', tstClassMet[clss]['accuracy'])
                print('Validation IoU: ', tstClassMet[clss]['iou'])
                print('Validation Dice: ', tstClassMet[clss]['dice'])
                print('Validation Precision: ', tstClassMet[clss]['precision'])
                print('Validation Recall: ', tstClassMet[clss]['recall'])
            

            if args.lrs == "true":
                scheduler.step(test_measure_mean)
            if prev_best_test == None or test_measure_mean > prev_best_test:
                print("Saving Best...")
                # save current model as temp best
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict()
                        if args.mgpu == "false"
                        else model.module.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": loss,
                        "test_measure_mean": test_measure_mean,
                        "test_measure_std": test_measure_std,
                    },
                    os.path.join(save_loc, "new_best.pt"),
                )
                # Delete old .PT
                if os.path.exists(os.path.join(save_loc, "best.pt")):
                    os.remove(os.path.join(save_loc, "best.pt"))
                # change name of old .PT
                os.rename(os.path.join(save_loc, "new_best.pt"), os.path.join(save_loc, "best.pt"))

                print('SAVED BEST')
                prev_best_test = test_measure_mean
            print("Saving last...")

            # save current model as temp best
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict()
                    if args.mgpu == "false"
                    else model.module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss,
                    "test_measure_mean": test_measure_mean,
                    "test_measure_std": test_measure_std,
                },
                os.path.join(save_loc, "new_last.pt"),
            )
            # Delete old .PT
            if os.path.exists(os.path.join(save_loc, "last.pt")):
                os.remove(os.path.join(save_loc, "last.pt"))
            # change name of old .PT
            os.rename(os.path.join(save_loc, "new_last.pt"), os.path.join(save_loc, "last.pt"))
            print('SAVED LAST')
    print("Finished Training")



def get_args():
    #set 
    parser = argparse.ArgumentParser(description="Train FCBFormer on specified dataset")
    parser.add_argument("--save-path", type=str, required=True)
    # parser.add_argument("--weight-path", type=str, required=True) # transformer pretrained weights (required for every run: even if model_weights are parsed)
    parser.add_argument("--model-weights", type=str, default='None') # load pretained weights from previous runs
    parser.add_argument("--no-ph-weights", type=str, default="True") # if True, does not load the predictor head (if you have a different shape output nodes)
    parser.add_argument("--data-root", type=str, required=True, dest="root")
    parser.add_argument("--tree-root", type=str, required=True) # full path to tree root file location
    parser.add_argument("--class-map", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs total. should be multiple of epochs per level, if not it will extend/reduce final level epochs")
    # parser.add_argument("--epochs-per-level", type=int, default=100, help="the number of epochs per hierarchical level. This should be adjusted to alow all hierarchical levels to be trained (epochs/epochs_per_level >= total_number_of_levels - 1).")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--val-dataset", type=str, default="None")
    parser.add_argument("--img-size", type=int, default=352)
    parser.add_argument("--learning-rate", type=str, default="[1e-4]", dest="lr") # 1e-4 1e-2
    parser.add_argument("--test-remove", type=str, default="True")
    parser.add_argument("--model-type", type=int, default=0, choices=[0, 1], help="0 for normal, 1 for hierarchical")
    parser.add_argument("--model-select", type=int, default=0, choices=[0, 1], help="0 for UNet, 1 for")
    parser.add_argument("--val-batch", type=int, default=1)
    parser.add_argument("--num-classes", type=list, default=[]) # dont specify this, it will be calculated from the class tree
    parser.add_argument("--num-workers", type=int, default=-1) #-1 for auto workers
    # parser.add_argument("--calc-super", type=str, default="False")
    parser.add_argument("--learning-rate-scheduler", type=str, default="true", dest="lrs")
    parser.add_argument("--learning-rate-scheduler-minimum", type=float, default=1e-6, dest="lrs_min") # 1e-6 1e-3
    parser.add_argument("--multi-gpu", type=str, default="false", dest="mgpu", choices=["true", "false"])
    parser.add_argument("--save-images-batch", type=str, default="False")
    parser.add_argument("--save-images-batch-num", type=int, default=10)
    # parser.add_argument("--hierarchical-loss", type=str, default="True") # REMOVE
    parser.add_argument("--inc-cross-val", type=str, default="True") 
    parser.add_argument("--folds", type=int, default=0) 
    parser.add_argument("--config", type=str, default='')
    parser.add_argument("--level-weights", type=str, default=None)
    parser.add_argument("--level0-pretrain-epochs", type=int, default=None)

    
    return parser.parse_args()


def main():

    args = get_args()

    if args.level0_pretrain_epochs is not None:
        args.level0_pretrain_epochs = int(args.level0_pretrain_epochs)

    if args.model_select == 1:
        update_config(config, args)
    args.inc_cross_val = True if args.inc_cross_val == "True" else False

    # converts level weights from string to list
    if args.level_weights is not None:
        args.level_weights = eval(args.level_weights)

    # check if save path, tree root, class map and model weights exist
    if not os.path.exists(args.save_path):
        args.save_path = os.path.join(os.curdir, args.save_path)
    if not os.path.exists(args.tree_root):
        args.tree_root = os.path.join(os.curdir, args.tree_root)
        if not os.path.exists(args.tree_root):
            print("Tree root does not exist")
            sys.exit()
    if not os.path.exists(args.class_map):
        args.class_map = os.path.join(os.curdir, args.class_map)
        if not os.path.exists(args.class_map):
            print("Class map does not exist")
            sys.exit()
    if args.model_weights != 'None' and not os.path.exists(args.model_weights):
        args.model_weights = os.path.join(os.curdir, args.model_weights)
        if not os.path.exists(args.model_weights):
            print("Model weights do not exist")
            sys.exit()
    if args.inc_cross_val:
        if not os.path.exists(args.root):
            args.root = os.path.join(os.curdir, args.root)
            if not os.path.exists(args.root):
                print("Root dataset does not exist")
                sys.exit()
        if not os.path.exists(args.val_dataset):
            args.val_dataset = os.path.join(os.curdir, args.val_dataset)
            if not os.path.exists(args.val_dataset):
                print("Validation dataset does not exist")
        train_locs = []
        val_locs = []
        for i in range(args.folds):
            train_locs.append(os.path.join(args.root, "fold_" + str(i+1), "train"))
            val_locs.append(os.path.join(args.root, "fold_" + str(i+1), "val"))

        args.root = train_locs
        args.val_dataset = val_locs
    else:
        if args.root != 'None' and not os.path.exists(args.val_dataset):
            if args.val_dataset == 'None':
                args.val_dataset = os.path.join(args.root, "val")
            else:
                args.val_dataset = os.path.join(os.curdir, args.val_dataset, "val")
            if not os.path.exists(args.val_dataset):
                print("Validation dataset does not exist")
        if args.root != 'None' and not os.path.exists(args.root):
            args.root = os.path.join(os.curdir, args.root)
            if not os.path.exists(args.root):
                print("Root dataset does not exist")
                sys.exit()
        args.root = [os.path.join(args.root, "train")]
        args.val_dataset = [args.val_dataset]
        
    args.lr = eval(args.lr)
    # converts string to boolean for some args
    args.no_ph_weights = True if args.no_ph_weights == "True" else False
    args.test_remove = True if args.test_remove == "True" else False
    # args.calc_super = True if args.calc_super == "True" else False
    args.save_images_batch = True if args.save_images_batch == "True" else False
    # args.hierarchical_loss = True if args.hierarchical_loss == "True" else False

    train(args)


if __name__ == "__main__":
    main()
    
