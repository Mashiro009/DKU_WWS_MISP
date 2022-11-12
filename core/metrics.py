# encoding: utf-8
"""
@author: Ming Cheng
@contact: ming.cheng@dukekunshan.edu.cn
"""
import os
import cv2
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt



def get_WWS_score(y_true, y_pred):
    FRR = get_false_reject_rate(y_true, y_pred)
    FAR = get_false_accept_rate(y_true, y_pred)
    score = FRR + FAR

    return score
    

def get_false_reject_rate(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise 'Not Equal Lenghts of y_true and y_pred'
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for true, pred in zip(y_true,y_pred):
        if true==1 and pred==1:
            TP += 1
        if true==0 and pred==0:
            TN += 1
        if true==1 and pred==0:
            FN += 1
        if true==0 and pred==1:
            FP += 1
         
    if TP+FN == 0:
        rate = 0.0
    else:
        rate = float(FN) / float(TP+FN)

    return rate


def get_false_accept_rate(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise 'Not Equal Lenghts of y_true and y_pred'  
    TP = 0
    TN = 0
    FP = 0
    FN = 0   
    for true, pred in zip(y_true,y_pred):
        if true==1 and pred==1:
            TP += 1
        if true==0 and pred==0:
            TN += 1
        if true==1 and pred==0:
            FN += 1
        if true==0 and pred==1:
            FP += 1
            
    if TN+FP == 0:
        rate = 0.0
    else:
        rate = float(FP) / float(TN+FP)

    return rate
    

def get_accuracy(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)

    return acc

def get_weighted_accuracy(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise 'Not Equal Lenghts of y_true and y_pred'  
    TP = 0
    TN = 0
    FP = 0
    FN = 0   
    for true, pred in zip(y_true,y_pred):
        if true==1 and pred==1:
            TP += 1
        if true==0 and pred==0:
            TN += 1
        if true==1 and pred==0:
            FN += 1
        if true==0 and pred==1:
            FP += 1
    
    pos_acc = TP / (TP + FN)
    neg_acc = TN / (TN + FP)
    weight = (TP + FN) / len(y_true)

    acc = float( pos_acc * (1-weight) + neg_acc * weight)

    return acc

    
def get_confusion_matrix(y_true, y_pred, labels):
    mat = confusion_matrix(y_true, y_pred, labels=range(len(labels)), normalize='true')
    
    return mat
    
     