#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 16:01:10 2021

@author: arpan

@Description: Finetuning Siamese I3D Pretrained for downstream task.
"""
import os
import sys
sys.path.insert(0, '../StrokeAttention')
sys.path.insert(0, '../../pytorch-i3d')
sys.path.insert(0, '../cluster_strokes')
sys.path.insert(0, '../cluster_strokes/lib')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from datasets.dataset import CricketStrokesDataset

import datasets.videotransforms as videotransforms
from torchvision import transforms
import numpy as np
import time
#import random

from utils import autoenc_utils
import siamese_net
import attn_utils
import copy
import pickle

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

pretrained_model_path = '/home/arpan/VisionWorkspace/pytorch-i3d/models/rgb_imagenet.pt'
selfsup_model_path = 'i3d_ep30_S16_SGD_B64Iter100.pt'  #'i3d_ep30_S16_SGD.pt'
#feat_path = "/home/arpan/VisionWorkspace/Cricket/CricketStrokeLocalizationBOVW/logs/bow_HL_ofAng_grid20"

VALID_ENDPOINTS = (
#        'Conv3d_1a_7x7',
#        'MaxPool3d_2a_3x3',
#        'Conv3d_2b_1x1',
#        'Conv3d_2c_3x3',
#        'MaxPool3d_3a_3x3',
#        'Mixed_3b',
#        'Mixed_3c',
#        'MaxPool3d_4a_3x3',
#        'Mixed_4b',
#        'Mixed_4c',
#        'Mixed_4d',
#        'Mixed_4e',
#        'Mixed_4f',
#        'MaxPool3d_5a_2x2',
        'Mixed_5b',
        'Mixed_5c',
        'Logits',
        'Predictions',
    )

def save_model_checkpoint(base_name, model, ep, opt):
    """
    Save the optimizer state with epoch no. along with the weights
    """
    # Save only the model params
    model_name = os.path.join(base_name, "i3dDown_ep"+str(ep)+"_"+opt+".pt")

    torch.save(model.state_dict(), model_name)
    print("Model saved to disk... : {}".format(model_name))

def load_weights(base_name, model, ep, opt):
    """
    Load the pretrained weights to the models' encoder and decoder modules
    """
    # Paths to encoder and decoder files
    model_name = os.path.join(base_name, "i3dDown_ep"+str(ep)+"_"+opt+".pt")
    if os.path.isfile(model_name):
        model.load_state_dict(torch.load(model_name))
        print("Loading I3D weights... : {}".format(model_name))
    return model

def train_model(model, dataloaders, criterion, optimizer, scheduler, labs_keys, 
                labs_values, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for bno, (inputs, vid_path, stroke, labels) in enumerate(dataloaders[phase]):
                # inputs of shape BATCH x SEQ_LEN x FEATURE_DIM
                labels = attn_utils.get_batch_labels(vid_path, stroke, labs_keys, labs_values, 1)
                # Extract spatio-temporal features from clip using 3D ResNet (For SL >= 16)
#                inputs = inputs.permute(0, 2, 1, 3, 4).float()
#                inputs[:, [0, 2], ...] = inputs[:, [2, 0], ...]       # convert RGB to BGR for C3D pretrained
                inputs = inputs.permute(0, 2, 1, 3, 4).float()
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()
                loss = 0
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    
                    logits = model.forward_once(inputs)
                    probs = F.softmax(logits.squeeze(axis=2), dim=1)
                    loss = criterion(probs, labels)
                    _, preds = torch.max(probs, 1)
                    
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item()
#                print("Iter : {} / {} :: Running Loss : {}".format(bno, 
#                      len(dataloaders[phase]), running_loss))
                running_corrects += torch.sum(preds == labels.data)
                
#                print("Batch No : {} / {}".format(bno, len(dataloaders[phase])))
#                if (bno+1) % 20 == 0:
#                    break
                    
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)  #(bno+1) 
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)  #(64*(bno+1)) 

            print('{} Loss: {:.6f} :: Acc: {:.6f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'test' and epoch_acc >= best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, \
          time_elapsed % 60))
    print('Best val Acc: {:6f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
    

def predict(model, dataloaders, labs_keys, labs_values, phase="val"):
    assert phase == "val" or phase=="test", "Incorrect Phase."
    model = model.eval()
    gt_list, pred_list, stroke_ids  = [], [], []
    # Iterate over data.
    for bno, (inputs, vid_path, stroke, labels) in enumerate(dataloaders[phase]):
        # inputs of shape BATCH x SEQ_LEN x FEATURE_DIM
        labels = attn_utils.get_batch_labels(vid_path, stroke, labs_keys, labs_values, 1)
        # Extract spatio-temporal features from clip using 3D ResNet (For SL >= 16)
        inputs = inputs.permute(0, 2, 1, 3, 4).float()
        inputs = inputs.to(device)
        # forward
        # track history if only in train
        with torch.set_grad_enabled(phase == 'train'):
            
            logits = model.forward_once(inputs)
            probs = F.softmax(logits.squeeze(axis=2), dim=1)
            gt_list.append(labels.tolist())
            pred_list.append((torch.max(probs, 1)[1]).tolist())
            for i, vid in enumerate(vid_path):
                stroke_ids.extend([vid+"_"+str(stroke[0][i].item())+"_"+str(stroke[1][i].item())] * 1)
        # statistics
#        running_loss += loss.item()
#                print("Iter : {} :: Running Loss : {}".format(bno, running_loss))
#                running_corrects += torch.sum(preds == labels.data)
        
        print("Batch No : {} / {}".format(bno, len(dataloaders[phase])))
#        if (bno+1) % 20 == 0:
#            break
#    epoch_loss = running_loss #/ len(dataloaders[phase].dataset)
#            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
    confusion_mat = np.zeros((model.i3d._num_classes, model.i3d._num_classes))
    gt_list = [g for batch_list in gt_list for g in batch_list]
    pred_list = [p for batch_list in pred_list for p in batch_list]
    prev_gt = stroke_ids[0]
    val_labels, pred_labels, vid_preds = [], [], []
    tm = 0
    for i, pr in enumerate(pred_list):
        if prev_gt != stroke_ids[i]:
            # find max category predicted in pred_labels
            val_labels.append(gt_list[i-1])
            pred_labels.append(max(set(vid_preds), key = vid_preds.count))
            print("Preds {} : {} :: {}".format(tm+1, vid_preds, pred_labels[-1]))
            print("GT {} : {}".format(tm+1, gt_list[i-1]))
            tm+=1
            vid_preds = []
            prev_gt = stroke_ids[i]
        vid_preds.append(pr)
        
    val_labels.append(gt_list[-1])
    pred_labels.append(max(set(vid_preds), key = vid_preds.count))
    correct = 0
    for i,true_val in enumerate(val_labels):
        if pred_labels[i] == true_val:
            correct+=1
        confusion_mat[pred_labels[i], true_val]+=1
    print('#'*30)
    print("GRU Sequence Classification Results:")
    print("%d/%d Correct" % (correct, len(pred_labels)))
    print("Accuracy = {} ".format( float(correct) / len(pred_labels)))
    print("Confusion matrix")
    print(confusion_mat)
    return (float(correct) / len(pred_labels))
    


def main(DATASET, LABELS, CLASS_IDS, BATCH_SIZE, ANNOTATION_FILE, SEQ_SIZE=16, STEP=16, 
         nstrokes=-1, N_EPOCHS=25, base_name=''):
    '''
    Extract sequence features from AutoEncoder.
    
    Parameters:
    -----------
    DATASET : str
        path to the video dataset
    LABELS : str
        path containing stroke labels
    CLASS_IDS : str
        path to txt file defining classes, similar to THUMOS
    BATCH_SIZE : int
        size for batch of clips
    SEQ_SIZE : int
        no. of frames in a clip (min. 16 for 3D CNN extraction)
    STEP : int
        stride for next example. If SEQ_SIZE=16, STEP=8, use frames (0, 15), (8, 23) ...
    partition : str
        'all' / 'train' / 'test' / 'val' : Videos to be considered
    nstrokes : int
        partial extraction of features (do not execute for entire dataset)
    
    Returns:
    --------
    trajectories, stroke_names
    
    '''
    ###########################################################################
    
    attn_utils.seed_everything(1234)
    
    if not os.path.isdir(base_name):
        os.makedirs(base_name)
    
    # Read the strokes 
    # Divide the highlight dataset files into training, validation and test sets
    train_lst, val_lst, test_lst = autoenc_utils.split_dataset_files(DATASET)
    print("No. of training videos : {}".format(len(train_lst)))
    
#    features, stroke_names_id = attn_utils.read_feats(feat_path, feat, snames)

#    ###########################################################################
#    
    ###########################################################################
    # Create a Dataset    
    # Clip level transform. Use this with framewiseTransform flag turned off
    train_transforms = transforms.Compose([videotransforms.RandomCrop(300),
                                           videotransforms.ToPILClip(),
                                           videotransforms.Resize((224, 224)),
                                           videotransforms.ToTensor(),
                                           videotransforms.Normalize(),
                                           videotransforms.ScaledNormMinMax(),
    ])
    test_transforms = transforms.Compose([videotransforms.CenterCrop(300), 
                                          videotransforms.ToPILClip(),
                                          videotransforms.Resize((224, 224)),
                                          videotransforms.ToTensor(),
                                          videotransforms.Normalize(),
                                         ])

    train_dataset = CricketStrokesDataset(train_lst, DATASET, LABELS, CLASS_IDS, 
                                         frames_per_clip=SEQ_SIZE, step_between_clips=STEP, 
                                         train=True, framewiseTransform=False, 
                                         transform=train_transforms)
    val_dataset = CricketStrokesDataset(val_lst, DATASET, LABELS, CLASS_IDS, 
                                        frames_per_clip=SEQ_SIZE, step_between_clips=STEP, 
                                        train=False, framewiseTransform=False, 
                                        transform=test_transforms)
    ###########################################################################
    
        
    labs_keys, labs_values = attn_utils.get_cluster_labels(ANNOTATION_FILE)
    
    num_classes = len(list(set(labs_values)))
    
    # created weighted Sampler for class imbalance
    if not os.path.isfile(os.path.join(base_name, "weights_c"+str(num_classes)+"_"+str(len(train_dataset))+".pkl")):
        samples_weight = attn_utils.get_sample_weights(train_dataset, labs_keys, labs_values, 
                                                       train_lst)
        with open(os.path.join(base_name, "weights_c"+str(num_classes)+"_"+str(len(train_dataset))+".pkl"), "wb") as fp:
            pickle.dump(samples_weight, fp)
    with open(os.path.join(base_name, "weights_c"+str(num_classes)+"_"+str(len(train_dataset))+".pkl"), "rb") as fp:
        samples_weight = pickle.load(fp)
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, 
                              sampler=sampler, worker_init_fn=np.random.seed(12),
                              num_workers=8)
    
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=8)
    
    data_loaders = {"train": train_loader, "test": val_loader}
    
    ###########################################################################  
        
    out_size = 200
#    model = siamese_net.SiameseVTransI3DNet(out_size, pretrained_model_path, SEQ_SIZE // 8)
    # load model and set loss function
    model = siamese_net.SiameseI3DNet(out_size, in_channels=3, 
                                      pretrained_wts=pretrained_model_path)
    print("Loading Self-sup pretrained wts : {}".format(os.path.join(base_name, 
          selfsup_model_path)))
    model.load_state_dict(torch.load(os.path.join(base_name, selfsup_model_path)))
    
    # reset the last layer (default requires_grad is True)
    for name, param in model.named_parameters():
        if not (True in [t in name for t in VALID_ENDPOINTS]):
            param.requires_grad = False
        print("\t {} --> {}".format(name, param.requires_grad))
    
#    inp_feat_size = model.i3d.fc.in_features
#    model.i3d.fc = nn.Linear(inp_feat_size, num_classes)
    
    model.i3d.replace_logits(num_classes)
    
    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)
    
    #    # Layers to finetune. Last layer should be displayed
    print("\nParams to learn:")
    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t {}".format(name))
    
    # better is SGD for I3D, with norm transform
    lr = 0.1
    optimizer_ft = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0000001)
#    optimizer_ft = torch.optim.Adam(params_to_update, lr=0.01)
    
    # Decay LR by a factor of 0.1 every 7 epochs
    lr_scheduler = StepLR(optimizer_ft, step_size=10, gamma=0.1)
    
#    lr = 5.0 # learning rate
#    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
#    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)    
    ###########################################################################
    # Training the model
    start = time.time()
    
    model = train_model(model, data_loaders, criterion, optimizer_ft, lr_scheduler, 
                        labs_keys, labs_values, num_epochs=N_EPOCHS)
    
    end = time.time()
    
    # save the best performing model
    save_model_checkpoint(base_name, model, N_EPOCHS, "SGD_Mixed5b5c_c5")
    # Load model checkpoints
    model = load_weights(base_name, model, N_EPOCHS, "SGD_Mixed5b5c_c5")
    
    print("Total Execution time for {} epoch : {}".format(N_EPOCHS, (end-start)))

    ###########################################################################
    
    print("Predicting ...")
    acc = predict(model, data_loaders, labs_keys, labs_values, phase='test')
    
    ###########################################################################
    
    # call count_paramters(model)  for displaying total no. of parameters
    print("#Parameters : {} ".format(autoenc_utils.count_parameters(model)))
    return acc


if __name__ == '__main__':
    # Local Paths
    LABELS = "/home/arpan/VisionWorkspace/Cricket/scripts/supporting_files/sample_set_labels/sample_labels_shots/ICC WT20"
    DATASET = "/home/arpan/VisionWorkspace/VideoData/sample_cricket/ICC WT20"
    CLASS_IDS = "/home/arpan/VisionWorkspace/Cricket/cluster_strokes/configs/Class Index_Strokes.txt"    
    ANNOTATION_FILE = "/home/arpan/VisionWorkspace/Cricket/CricketStrokeLocalizationBOVW/shots_classes.txt"
    base_path = "/home/arpan/VisionWorkspace/Cricket/StrokeSelfSupervision/logs/siami3d_selfsup"
    
    seq_sizes = range(16, 17, 1)
    STEP = 4
    BATCH_SIZE = 64
    N_EPOCHS = 30
    
    attn_utils.seed_everything(1234)
    acc = []

    print("Finetuning I3D pretrained on pretext task for downstream action recognition...")
    print("EPOCHS = {} ".format(N_EPOCHS))
    for SEQ_SIZE in seq_sizes:
        print("SEQ_SIZE : {}".format(SEQ_SIZE))
        acc.append(main(DATASET, LABELS, CLASS_IDS, BATCH_SIZE, ANNOTATION_FILE, SEQ_SIZE,
                        STEP, nstrokes=-1, N_EPOCHS=N_EPOCHS, base_name=base_path))
        
        
    print("*"*60)
    print("SEQ_SIZES : {}".format(seq_sizes))
    print("Accuracy values : {}".format(acc))



