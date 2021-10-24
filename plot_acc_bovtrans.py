#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 06:25:01 2020

@author: arpan
"""

from matplotlib import pyplot as plt
import os
import numpy as np
import re
#plt.style.use('ggplot')


def plot_traintest_loss(keys, l, xlab, ylab, seq, batch, destfile):
    # Plot the loss values for the different epochs in one trained model
    keylist = range(1, len(l[keys[0]])+1)      # x-axis for 30 epochs
    cols = ['r','g','b', 'c']    
    print("Iteration and Accuracy Lists : ")
    print(keylist)
    print(l)
    fig = plt.figure(2)
    plt.title("Loss Vs Epoch (Seq_Len="+str(seq)+", Batch="+str(batch)+")", fontsize=12)
    plt.plot(keylist, l[keys[0]], lw=1, color=cols[0], marker='.', label= keys[0])
    plt.plot(keylist, l[keys[1]], lw=1, color=cols[1], marker='.', label= keys[1])
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.legend()
#    plt.show()
    plt.savefig(destfile, bbox_inches='tight', dpi=300)    
    plt.close(fig)
    return

def plot_traintest_accuracy(keys, l, xlab, ylab, seq, batch, best, destfile):
    # Plot the accuracy values for the different epochs in one trained model
    keylist = range(1, len(l[keys[0]])+1)      # x-axis for 30 epochs
    cols = ['r','g','b', 'c']
    print("Iteration and Accuracy Lists : ")
    print(keylist)
    print(l)
    fig = plt.figure(2)
    plt.title("Accuracy Vs Epoch (Seq_Len="+str(seq)+", Batch="+str(batch)+")", fontsize=12)
    plt.plot(keylist, l[keys[0]], lw=1, color=cols[0], marker='.', label= keys[0])
    plt.plot(keylist, l[keys[1]], lw=1, color=cols[1], marker='.', label= keys[1])
    plt.xlabel(xlab)
    plt.ylabel(ylab)
#    plt.axvline(x=best, color='r', linestyle='--')
    plt.legend()
    plt.ylim(bottom=0, top=1)
#    plt.show()
    plt.savefig(destfile, bbox_inches='tight', dpi=300)
    plt.close(fig)
    return

def plot_accuracy(x, keys, l, xlab, ylab, fname):
    
#    keys = ["HOG", "HOOF", "OF Grid 20", "C3D $\mathit{FC7}$: $w_{c3d}=17$"]
#    l = {keys[0]: hog_acc, keys[1]: hoof_acc, keys[2]: of30_acc, keys[3]:accuracy_17_30ep}
    cols = ['r','g','b', 'c']        
    print(l)
    fig = plt.figure(2)
    plt.title("Accuracy Vs #Words", fontsize=12)
    for i in range(len(keys)):
        acc = l[keys[i]]
        plt.plot(x[(len(x)-len(acc)):], acc, lw=1, color=cols[i], marker='.', label=keys[i])
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.legend(loc=4)
    plt.ylim(bottom=0, top=1)
    plt.show()
    fig.savefig(fname, bbox_inches='tight', dpi=300)
    plt.close(fig)
    return

def plot_HA_acc_of20(x, keys, l, xlab, ylab, fname):    
    cols = ['r','g','b', 'c']
    print(l)
    fig = plt.figure(2)
    plt.title("Accuracy Vs Sequence Length", fontsize=13)
    for i in range(len(keys)):
        acc = l[keys[i]]
        plt.plot(x[(len(x)-len(acc)):], acc, lw=1, color=cols[i], marker='.', label=keys[i])
    plt.xlabel(xlab, fontsize=12)
    plt.ylabel(ylab, fontsize=12)
    plt.axvline(x=18, color='b', linestyle='--')
    plt.legend(loc=4)
    plt.ylim(bottom=0, top=1)
    plt.show()
    fig.savefig(fname, bbox_inches='tight', dpi=300)
    plt.close(fig)
    return

def plot_HA_SA_Raw_acc_of20(x, keys, l, xlab, ylab, fname):
    cols = ['r','b', 'g', 'c']
    print(l)
    fig = plt.figure(2)
    plt.title("Accuracy Vs Sequence Length", fontsize=13)
    for i in range(len(keys)):
        acc = l[keys[i]]
        plt.plot(x[(len(x)-len(acc)):], acc, lw=1, color=cols[i], marker='.', label=keys[i])
    plt.xlabel(xlab, fontsize=12)
    plt.ylabel(ylab, fontsize=12)
#    plt.axvline(x=24, color='b', linestyle='--')
    plt.legend(loc=4)
    plt.ylim(bottom=0, top=1)
    plt.show()
    fig.savefig(fname, bbox_inches='tight', dpi=300)
    plt.close(fig)
    return

def plot_S2S_HA_SA_acc_of20(x, keys, l, xlab, ylab, fname):
    cols = ['r','g','b', 'c']
    print(l)
    fig = plt.figure(2)
    plt.title("Accuracy Vs Sequence Length", fontsize=13)
    for i in range(len(keys)):
        acc = l[keys[i]]
        plt.plot(x[(len(x)-len(acc)):], acc, lw=1, color=cols[i], marker='.', label=keys[i])
    plt.xlabel(xlab, fontsize=12)
    plt.ylabel(ylab, fontsize=12)
    plt.axvline(x=24, color='r', linestyle='--')
#    plt.axvline(x=24, color='b', linestyle='--')
    plt.legend(loc=3)
    plt.ylim(bottom=0, top=1)
    plt.show()
    fig.savefig(fname, bbox_inches='tight', dpi=300)
    plt.close(fig)
    return

def plot_S2S_Vs_Siamese_acc_of20(x, keys, l, xlab, ylab, fname):
    cols = ['r','g','b', 'c']
    print(l)
    fig = plt.figure(2)
    plt.title("Accuracy Vs Sequence Length", fontsize=13)
    for i in range(len(keys)):
        acc = l[keys[i]]
        plt.plot(x[(len(x)-len(acc)):], acc, lw=1, color=cols[i], marker='.', label=keys[i])
    plt.xlabel(xlab, fontsize=12)
    plt.ylabel(ylab, fontsize=12)
#    plt.axvline(x=24, color='r', linestyle='--')
#    plt.axvline(x=24, color='b', linestyle='--')
    plt.legend(loc=4)
    plt.ylim(bottom=0, top=1)
    plt.show()
    fig.savefig(fname, bbox_inches='tight', dpi=300)
    plt.close(fig)
    return

def plot_acc_of20_GRU_HA(x, keys, l, xlab, ylab, fname):
    cols = ['r','g','b', 'c']
    print(l)
    fig = plt.figure(2)
    plt.title("Accuracy Vs No. of Clusters(C)", fontsize=13)
    for i in range(len(keys)):
        acc = l[keys[i]]
        plt.plot(x[(len(x)-len(acc)):], acc, lw=1, color=cols[i], marker='.', label=keys[i])
    plt.xlabel(xlab, fontsize=12)
    plt.ylabel(ylab, fontsize=12)
    plt.legend(loc=4)
    plt.ylim(bottom=0, top=1)
    plt.show()
    fig.savefig(fname, bbox_inches='tight', dpi=300)
    plt.close(fig)
    return

def plot_acc_diff_feats(x, keys, l, xlab, ylab, fname):
    cols = ['r','g','b', 'c']
    print(l)
    fig = plt.figure(2)
    plt.title("Accuracy Vs Sequence Length", fontsize=13)
    for i in range(len(keys)):
        acc = l[keys[i]]
        plt.plot(x[(len(x)-len(acc)):], acc, lw=1, color=cols[i], marker='.', label=keys[i])
    plt.xlabel(xlab, fontsize=12)
    plt.ylabel(ylab, fontsize=12)
    plt.axvline(x=20, color='r', linestyle='--')
    plt.axvline(x=2, color='g', linestyle='--')
    plt.axvline(x=18, color='b', linestyle='--')     # old x = 2
    plt.axvline(x=16, color='c', linestyle='--')  # old x=32
    plt.legend(loc=4)
    plt.ylim(bottom=0, top=1)
    plt.show()
    fig.savefig(fname, bbox_inches='tight', dpi=300)
    plt.close(fig)
    return

def plot_sample_dist():
    x = ['C-1', 'C-2', 'C-3', 'C-4', 'C-5']
    cat_wts = [2644.0, 14330.0, 7837.0, 3926.0, 9522.0]
    
    x_pos = [i for i, _ in enumerate(x)]
    
    plt.bar(x_pos, cat_wts, width=0.5)#, color='cyan')
    plt.xlabel("Stroke Categories", fontsize=12)
    plt.ylabel("#Samples", fontsize=12)
    plt.title("Number of samples per category", fontsize=16)
    plt.xticks(x_pos, x)
    plt.savefig("sampleDist.png", dpi=300)
    plt.show()


if __name__ == '__main__':
    
    seq40 = list(range(2, 41, 2))
    
    ###########################################################################
    # Fully Supervised Approaches. Feed class labels for loss calculation
    # StrokeSelfSupervised/logs/bovtrans/HA_of20_Hidden200_C1000/log_*
    # Check test accuracy for seq = 26, Acc = 0.7452830188679245
    OF20_HA_C1k = [0.6857142857142857, 0.6476190476190476, 0.7238095238095238, 
                   0.6761904761904762, 0.7142857142857143, 0.6952380952380952, 
                   0.7238095238095238, 0.7047619047619048, 0.7047619047619048, 
                   0.6952380952380952, 0.7333333333333333, 0.7333333333333333, 
                   0.7619047619047619, 0.7047619047619048, 0.6857142857142857, 
                   0.6857142857142857, 0.7047619047619048, 0.7333333333333333, 
                   0.7238095238095238, 0.7238095238095238]
    
    # StrokeSelfSupervision/logs/bovtrans/HA_OF20_Hidden200_C200/log_*
    # Check test accuracy for seq = 36, Acc = 0.7169811320754716
    OF20_HA_C200 = [0.5904761904761905, 0.580952380952381, 0.6571428571428571, 
                    0.6666666666666666, 0.6095238095238096, 0.6952380952380952, 
                    0.7047619047619048, 0.6761904761904762, 0.7428571428571429, 
                    0.6095238095238096, 0.7142857142857143, 0.7523809523809524, 
                    0.7333333333333333, 0.6666666666666666, 0.7333333333333333, 
                    0.7428571428571429, 0.7238095238095238, 0.7619047619047619, 
                    0.7047619047619048, 0.7142857142857143]
    
    # Check test accuracy for seq = 18, Acc = 0.7735849056603774
    OF20_HA_C300 = [0.6476190476190476, 0.6761904761904762, 0.6761904761904762, 
                    0.7047619047619048, 0.7238095238095238, 0.7333333333333333, 
                    0.7333333333333333, 0.7047619047619048, 0.7714285714285715, 
                    0.7619047619047619, 0.7142857142857143, 0.7619047619047619, 
                    0.7142857142857143, 0.7238095238095238, 0.7047619047619048, 
                    0.7523809523809524, 0.7047619047619048, 0.7142857142857143, 
                    0.7238095238095238, 0.7428571428571429]
    
    OFMagAng20_HA_C200 = [0.4, 0.3904761904761905, 0.42857142857142855, 
                          0.42857142857142855, 0.45714285714285713, 0.5333333333333333, 
                          0.4380952380952381, 0.49523809523809526, 0.5428571428571428, 
                          0.5333333333333333, 0.5333333333333333, 0.6, 0.6, 
                          0.6666666666666666, 0.6095238095238096, 0.6, 0.638095238095238, 
                          0.580952380952381, 0.6190476190476191, 0.6095238095238096]
    
    OFMagAng20_HA_C300 = [0.41904761904761906, 0.4, 0.41904761904761906, 0.5523809523809524, 
                          0.45714285714285713, 0.4095238095238095, 0.5619047619047619, 
                          0.6190476190476191, 0.5428571428571428, 0.5238095238095238, 
                          0.6, 0.6285714285714286, 0.5523809523809524, 0.5523809523809524, 
                          0.638095238095238, 0.5904761904761905, 0.6952380952380952, 
                          0.5714285714285714, 0.6095238095238096, 0.5904761904761905]
    
    # Check test accuracy for seq = 22, Acc =  0.7169811320754716 
    OF20_SA_C200 = [0.6285714285714286, 0.6666666666666666, 0.6857142857142857, 
                    0.6666666666666666, 0.7142857142857143, 0.5904761904761905, 
                    0.6761904761904762, 0.6857142857142857, 0.6285714285714286, 
                    0.7333333333333333, 0.7523809523809524, 0.7238095238095238, 
                    0.7238095238095238, 0.7428571428571429, 0.7428571428571429, 
                    0.6952380952380952, 0.7333333333333333, 0.7142857142857143, 
                    0.7333333333333333, 0.6666666666666666]
    
    # Check test accuracy for seq = 20, Acc =  0.7735849056603774
    OF20_SA_C300 = [0.6, 0.47619047619047616, 0.7333333333333333, 0.6095238095238096, 
                    0.6857142857142857, 0.7047619047619048, 0.6952380952380952, 
                    0.7047619047619048, 0.6666666666666666, 0.7619047619047619, 
                    0.7619047619047619, 0.7428571428571429, 0.7333333333333333, 
                    0.7333333333333333, 0.7047619047619048, 0.6952380952380952, 
                    0.6476190476190476, 0.6952380952380952, 0.7333333333333333, 
                    0.6666666666666666]
    
    # check test accuracy for seq = 40, Acc = 0.6792452830188679
    OF20_raw = [0.638095238095238, 0.5333333333333333, 0.6476190476190476, 
                0.638095238095238, 0.5714285714285714, 0.6857142857142857, 
                0.6476190476190476, 0.5428571428571428, 0.5714285714285714, 
                0.5619047619047619, 0.6190476190476191, 0.6, 0.6666666666666666, 
                0.6285714285714286, 0.638095238095238, 0.6, 0.5714285714285714, 
                0.5523809523809524, 0.6761904761904762, 0.6952380952380952]

    # check test accuracy for seq = 30, Acc = 0.7169811320754716
    OF20_HA_C300_FineLast = [0.6952380952380952, 0.7238095238095238, 0.7142857142857143, 
                             0.7142857142857143, 0.6857142857142857, 0.6857142857142857, 
                             0.7142857142857143, 0.7047619047619048, 0.6761904761904762, 
                             0.7047619047619048, 0.7142857142857143, 0.7238095238095238, 
                             0.7142857142857143, 0.7047619047619048, 0.7428571428571429,
                             0.6952380952380952, 0.6952380952380952, 0.6952380952380952, 
                             0.7142857142857143, 0.6857142857142857]
    
    ###########################################################################
    ###########################################################################
    
    
    # Self-Sup : Use word sequences for future step = 1 for loss calculation
    # Seq2Seq Methods : StrokeSelfSupervision/logs/bovtrans_seq2seq/
    # Extract from log file
    OF20_HA_S2S_C300_Seq30Acc = [0.5, 0.5, 0.5, 0.5, 0.5, 
                        0.5, 0.5, 0.5, 0.5, 0.5, 
                        0.5, 0.5, 0.5, 0.5, 0.5]
    
    OF20_SA_S2S_C300_Seq30Acc = [0.5, 0.5, 0.5, 0.5, 0.5, 
                        0.5, 0.5, 0.5, 0.5, 0.5, 
                        0.5, 0.5, 0.5, 0.5, 0.5]
    
    #OF20_raw = []
    # Downstream
    
    # Test Acc=0.7547169811320755 for Seq=24  
    # Test Acc = 0.7924528301886793  for Seq=38
    OF20_HA_S2SDown_C300 = [0.6476190476190476, 0.6857142857142857, 0.6666666666666666, 
                            0.7142857142857143, 0.6952380952380952, 0.6857142857142857, 
                            0.7142857142857143, 0.7333333333333333, 0.7333333333333333, 
                            0.7428571428571429, 0.7238095238095238, 0.7714285714285715, 
                            0.7619047619047619, 0.7523809523809524, 0.7428571428571429, 
                            0.7523809523809524, 0.7047619047619048, 0.7142857142857143, 
                            0.7619047619047619, 0.7238095238095238]
    
    # Downstream
    # Test Acc=0.7735849056603774    for Seq=38
    OF20_SA_S2SDown_C300 = [0.6476190476190476, 0.6285714285714286, 0.6285714285714286, 
                            0.6761904761904762, 0.6190476190476191, 0.6476190476190476, 
                            0.6571428571428571, 0.6476190476190476, 0.6571428571428571, 
                            0.6761904761904762, 0.6952380952380952, 0.6571428571428571, 
                            0.7333333333333333, 0.7142857142857143, 0.7238095238095238, 
                            0.7047619047619048, 0.7142857142857143, 0.7428571428571429, 
                            0.7619047619047619, 0.7333333333333333]
    
    
    ###########################################################################
    ###########################################################################
    # Contrastive Methods :  Results not good
    
    # Downstream
    # Test Acc = 0.6886792452830188,   for Seq = 16
    OF20_HA_SSDown_C300 = [0.4857142857142857, 0.4857142857142857, 0.5047619047619047, 
                           0.5047619047619047, 0.5142857142857142, 0.49523809523809526, 
                           0.4666666666666667, 0.5238095238095238, 0.47619047619047616, 
                           0.4666666666666667, 0.4857142857142857, 0.5047619047619047, 
                           0.45714285714285713, 0.4666666666666667, 0.45714285714285713, 
                           0.4666666666666667, 0.45714285714285713, 0.47619047619047616, 
                           0.47619047619047616, 0.47619047619047616]
    
    # Test Acc = 0.5660377358490566  ,  for Seq = 10
    OF20_SA_SSDown_C300 = [0.37142857142857144, 0.5142857142857142, 0.3523809523809524, 
                           0.4380952380952381, 0.5333333333333333, 0.44761904761904764, 
                           0.5047619047619047, 0.5238095238095238, 0.49523809523809526, 
                           0.5238095238095238, 0.49523809523809526, 0.5142857142857142, 
                           0.4380952380952381, 0.47619047619047616, 0.45714285714285713, 
                           0.5238095238095238, 0.4857142857142857, 0.49523809523809526, 
                           0.47619047619047616, 0.5047619047619047]
    
    
    ###########################################################################
    ###########################################################################
#    # Contrastive Methods :  Wrong results: Pretrained Selfsup wts were not loaded
#    
#    # Downstream
#    # Test Acc =  0.7264150943396226,   for Seq = 26
#    OF20_HA_SSDown_C300 = [0.6952380952380952, 0.7047619047619048, 0.6285714285714286, 
#                           0.7142857142857143, 0.6761904761904762, 0.6857142857142857, 
#                           0.6952380952380952, 0.6952380952380952, 0.7142857142857143, 
#                           0.6857142857142857, 0.7142857142857143, 0.6952380952380952, 
#                           0.7428571428571429, 0.7047619047619048, 0.6952380952380952, 
#                           0.7142857142857143, 0.7333333333333333, 0.6571428571428571, 
#                           0.7238095238095238, 0.7142857142857143]
#    
#    # Test Acc =  0.7924528301886793 ,  for Seq = 26
#    OF20_SA_SSDown_C300 = [0.580952380952381, 0.5428571428571428, 0.5238095238095238, 
#                           0.6571428571428571, 0.6476190476190476, 0.6761904761904762, 
#                           0.5619047619047619, 0.6666666666666666, 0.638095238095238, 
#                           0.6857142857142857, 0.6857142857142857, 0.6857142857142857, 
#                           0.7142857142857143, 0.6666666666666666, 0.6476190476190476, 
#                           0.6095238095238096, 0.6666666666666666, 0.6761904761904762, 
#                           0.6761904761904762, 0.6857142857142857]
    

    ###########################################################################
    # Plot OF20 HA Comparisons for C=1k, 200, 300 
#    
#    keys = ["OF20 HA ; C=1000", "OF20 HA ; C=200", "OF20 HA ; C=300"]  # OFGrid20 Hidden=256
#    l = {keys[0] : OF20_HA_C1k, keys[1] : OF20_HA_C200, keys[2] : OF20_HA_C300}
#    
#    fname = os.path.join("logs", "OF20_HA200_HA300_HA1000.png")
#    plot_HA_acc_of20(seq40, keys, l, "Sequence Length", "Accuracy", fname)

    ###########################################################################
#    # Plot OF20 Raw, HA and SA Accuracies for C=300 
#    
#    keys = ["OF20 HA; C=300", "OF20 SA; C=300", "OF20 Raw"]  # 
#    l = {keys[0] : OF20_HA_C300, keys[1]: OF20_SA_C300, keys[2] : OF20_raw}
#    
#    fname = os.path.join("logs", "OF20_HA_Vs_SA_Vs_Raw.png")
#    plot_HA_SA_Raw_acc_of20(seq40, keys, l, "Sequence Length", "Accuracy", fname)
#    
    ###########################################################################
    # Plot OF20 MagAng, and HA Accuracies for C={200, 300}
#    
#    keys = ["OF20 Ang HA; C=200", "OF20 Ang HA; C=300", "OF20 Mag-Ang HA; C=200", 
#            "OF20 Mag-Ang HA; C=300"]
#    l = {keys[0] : OF20_HA_C200, keys[1]: OF20_HA_C300, keys[2] : OFMagAng20_HA_C200,
#         keys[3] : OFMagAng20_HA_C300}
#    
#    fname = os.path.join("logs", "OF20HA_Ang_Vs_MagAng.png")
#    plot_HA_SA_Raw_acc_of20(seq40, keys, l, "Sequence Length", "Accuracy", fname)
    
    ###########################################################################
#    # Compare Random Wts Model (Finetune Last Layer) Vs Finetune all layers
#    
#    keys = ["OF20 HA; C=300; Last Layer", "OF20 HA; C=300; All Layers"]
#    l = {keys[1] : OF20_HA_C300, keys[0]: OF20_HA_C300_FineLast}
#    
#    fname = os.path.join("logs", "OF20HA_FullModel_Vs_LastLayer.png")
#    plot_HA_SA_Raw_acc_of20(seq40, keys, l, "Sequence Length", "Accuracy", fname)   
    
    ###########################################################################
#    # Compare S2S downstream accuracy with supervised  
#    
#    keys = ["S2S HA; C=300", "S2S SA; C=300", "Sup HA; C=300", "Sup SA; C=300", ]
##            "OF20 Mag-Ang HA; C=300"]
#    l = {keys[0] : OF20_HA_S2SDown_C300, keys[1]: OF20_SA_S2SDown_C300, 
#         keys[2] : OF20_HA_C300, keys[3] : OF20_SA_C300,}
##         keys[3] : }
#    
#    fname = os.path.join("logs", "S2SHA_S2SSA_SupHA_SupSA.png")
#    plot_S2S_HA_SA_acc_of20(seq40, keys, l, "Sequence Length", "Accuracy", fname)
    ###########################################################################
    # Compare S2S with SS (Siamese)
    keys = ["S2S HA; C=300", "S2S SA; C=300", "Siam HA; C=300", "Siam SA; C=300"]
#            "OF20 Mag-Ang HA; C=300"]
    l = {keys[0] : OF20_HA_S2SDown_C300, keys[1]: OF20_SA_S2SDown_C300, 
         keys[2] : OF20_HA_SSDown_C300, keys[3]: OF20_SA_SSDown_C300}
#         keys[3] : }
    
    fname = os.path.join("logs", "S2S_Vs_Siamese.png")
    plot_S2S_Vs_Siamese_acc_of20(seq40, keys, l, "Sequence Length", "Accuracy", fname)
 
    
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    # Plot Comparison for different features with HA and C=1k
    
#    keys = ["OF Grid=20", "HOOF Bins=20", "2DCNN", "3DCNN"]  # Hidden=256, C=1k 
#    l = {keys[0] : OF20_HA_C1k, keys[1] : HOOF_B20_HA, 
#         keys[2] : CNN2D_HA_C1k, keys[3] : CNN3D_HA_C1k}
#    
#    fname = os.path.join("logs", "CompareFeats_v1.png")
#    plot_acc_diff_feats(seq, keys, l, "Sequence Length", "Accuracy", fname)
    
    ###########################################################################
    
    ###########################################################################
    
#    ###########################################################################
#    # 2 Stream (OF20 + HOG SA C=1000) validation accuracies for seq=range(2, 41, 2)
#    nclust_acc = [0.5428571428571428, 0.7428571428571429, 0.6, 0.7142857142857143, 
#                  0.7238095238095238, 0.7047619047619048, 0.7142857142857143, 
#                  0.6857142857142857, 0.7238095238095238, 0.780952380952381, 
#                  0.7428571428571429, 0.780952380952381, 0.7428571428571429, 
#                  0.7428571428571429, 0.7333333333333333, 0.7428571428571429, 
#                  0.12380952380952381, 0.7047619047619048, 0.7142857142857143, 
#                  0.638095238095238]
#    
#    keys = ["2 Stream GRU Hidden=256"]
#    l = {keys[0] : nclust_acc}
#    fname = os.path.join("logs/plot_data", "2Stream_OF20_HOG_seq2_40.png")
#    plot_acc_of20(list(range(2, 41, 2)), keys, l, "Seq. Length", "Accuracy", fname)
#    
    ###########################################################################
#    
#    # Plot the Seq2Seq HA training on OF20 feats with Seq=30 and C=1000
#    # Val Accuracy : 0.780952380952381 
#    file = "logs/bovtrans_seq2seq/HA_of20_Hidden200_C300_SSFuturePred/log_plot_seq30_F1.txt" # 
##    file = "logs/bovtrans_selfsup/HA_of20_Hidden200_C300_HardF1/log_siamtrans_Adam_lr0.001_ep60_step20_seq30_step4_F1.txt"
#    train_loss, test_loss, train_acc, test_acc = [], [], [], []
#    with open(file, 'r') as fp:
#        lines = fp.readlines()
#    for line in lines: 
#        line = line.strip()
#        if 'train Loss:' in line:
#            t = re.findall("\d+\.\d+", line)
#            train_loss.append(float(t[0]))
#            train_acc.append(float(t[1]))
#        elif 'test Loss:' in line:
#            t = re.findall("\d+\.\d+", line)
#            test_loss.append(float(t[0]))
#            test_acc.append(float(t[1]))
#        elif 'SEQ_SIZE : ' in line:
#            t = re.findall("\d+", line)
#            seq = str(t[0])
#    
#    l1 = {"train loss" : train_loss[1:], "test loss": test_loss[1:]}
#    l2 = {"train accuracy" : train_acc, "test accuracy" : test_acc}
#    best_ep = test_acc.index(max(test_acc)) + 1
#    loss_file = 'logs/seq2seqHAseq30F1_losses_seq'+str(seq)+'.png'
#    acc_file = 'logs/seq2seqHAseq30F1_acc_seq'+str(seq)+'.png'
#    plot_traintest_loss(["train loss", "test loss"], l1, "Epochs", "Loss", seq, 32, loss_file)
#    plot_traintest_accuracy(["train accuracy", "test accuracy"], l2, "Epochs", "Accuracy", seq, 
#                            32, best_ep, acc_file)
#    
#    ###########################################################################
#    
#    # Plot the Seq2Seq HA training accuracy on OF20 feats with diff fstep for Seq=30 and C=300
#    file = "logs/bovtrans_seq2seq/HA_of20_Hidden200_C300_SSFuturePred/log_plot_seq30_F1.txt" # 
##    file = "logs/bovtrans_selfsup/HA_of20_Hidden200_C300_HardF1/log_siamtrans_Adam_lr0.001_ep60_step20_seq30_step4_F1.txt"
#    train_loss, test_loss, train_acc, test_acc = [], [], [], []
#    with open(file, 'r') as fp:
#        lines = fp.readlines()
#    for line in lines: 
#        line = line.strip()
#        if 'train Loss:' in line:
#            t = re.findall("\d+\.\d+", line)
#            train_loss.append(float(t[0]))
#            train_acc.append(float(t[1]))
#        elif 'test Loss:' in line:
#            t = re.findall("\d+\.\d+", line)
#            test_loss.append(float(t[0]))
#            test_acc.append(float(t[1]))
#        elif 'SEQ_SIZE : ' in line:
#            t = re.findall("\d+", line)
#            seq = str(t[0])
#    
#    l1 = {"train loss" : train_loss[1:], "test loss": test_loss[1:]}
#    l2 = {"train accuracy" : train_acc, "test accuracy" : test_acc}
#    best_ep = test_acc.index(max(test_acc)) + 1
#    loss_file = 'logs/seq2seqHAseq30F1_losses_seq'+str(seq)+'.png'
#    acc_file = 'logs/seq2seqHAseq30F1_acc_seq'+str(seq)+'.png'
#    plot_traintest_loss(["train loss", "test loss"], l1, "Epochs", "Loss", seq, 32, loss_file)
#    plot_traintest_accuracy(["train accuracy", "test accuracy"], l2, "Epochs", "Accuracy", seq, 
#                            32, best_ep, acc_file)
    
    ###########################################################################
#    # Plot the C3D finetuning losses (SEQ_SIZE = 16, STEP = 4, BATCH = 16, ITer=150/Ep)
##    file = "logs/plot_data/C3DFine_seq16_SGD.txt" # 
#    file = "logs/plot_data/C3DFine_seq16_SGD_newTransforms.txt"
#    train_loss, test_loss, train_acc, test_acc = [], [], [], []
#    seq = 16
#    with open(file, 'r') as fp:
#        lines = fp.readlines()
#    for line in lines: 
#        line = line.strip()
#        if 'train Loss:' in line:
#            t = re.findall("\d+\.\d+", line)
#            train_loss.append(float(t[0]))
#            train_acc.append(float(t[1]))
#        elif 'test Loss:' in line:
#            t = re.findall("\d+\.\d+", line)
#            test_loss.append(float(t[0]))
#            test_acc.append(float(t[1]))
#            
#    l1 = {"train loss" : train_loss, "test loss": test_loss}
#    l2 = {"train accuracy" : train_acc, "test accuracy" : test_acc}
#    best_ep = test_acc.index(max(test_acc)) + 1
#    loss_file = 'logs/plot_data/C3DFine_seq16_newTrans.png'
#    acc_file = 'logs/plot_data/C3DFine_acc_seq16_newTrans.png'
#    plot_traintest_loss(["train loss", "test loss"], l1, "Epochs", "Loss", seq, 16, loss_file)
#    plot_traintest_accuracy(["train accuracy", "test accuracy"], l2, "Epochs", "Accuracy", seq, 
#                            16, best_ep, acc_file)
    
    ###########################################################################
    