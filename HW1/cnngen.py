# -*- coding: utf-8 -*-
"""Untitled1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1azZVdaXpqTaniyIwn2gVpU--nT8fq3Mg
"""

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

import sys
sys.path.append("../src/")
import keras_cifar_cnn as cnn
import tf_optimizer as tf_opt
import callbacks as clb
import utils as ut
from plotting import set_display_settings
set_display_settings()

from keras.optimizers import TFOptimizer
from keras.callbacks import History

x_train, y_train, validation_set, datagen = cnn.get_dataset()

model = cnn.get_cnn()
batch_size = 32

model = cnn.get_cnn()
metrics_history = [clb.AggregateMetricsOnBatchEnd(), clb.AggregateMetricsOnEpochEnd()]

for lr in [0.001, 0.0005, 0.0001, 0.00001]:        
    # max norm was working fine but a little worse than Adam, I have tried std norm.
    #optimizer = TFOptimizer(tf_opt.LmaxNormalizedSGD(lr=lr))
    optimizer = TFOptimizer(tf_opt.AdaptiveNormalizedSGD(
        lr=lr, lr_update=0.0, momentum=0.0, momentum_update=0.00, norm_type='std'
    ))
    model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(
        x=x_train, y=y_train, batch_size=batch_size,
        epochs=15,
        validation_data=validation_set, 
        callbacks=metrics_history)

hist = metrics_history[1].monitor_values['accuracies']
for k, v in hist.items():
    if len(v) != 0: plt.plot(v, label=k)
plt.legend()

from keras.optimizers import Adam, SGD

model = cnn.get_cnn()
adam_metrics_history = [clb.AggregateMetricsOnBatchEnd(), clb.AggregateMetricsOnEpochEnd()]

for lr in [0.001, 0.0005, 0.0001]:    
    optimizer = Adam(lr=lr)
    model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(
        x=x_train, y=y_train, batch_size=batch_size,
        epochs=15,
        validation_data=validation_set, 
        callbacks=adam_metrics_history)

hist = adam_metrics_history[1].monitor_values['accuracies']
for k, v in hist.items():
    if len(v) != 0: plt.plot(v, label=k)
plt.legend()

model = cnn.get_cnn()
momentum_metrics_history = [clb.AggregateMetricsOnBatchEnd(), clb.AggregateMetricsOnEpochEnd()]

for lr in [0.01, 0.005, 0.001]:    
    optimizer = SGD(lr=lr, momentum=0.9)
    model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(
        x=x_train, y=y_train, batch_size=batch_size,
        epochs=15,
        validation_data=validation_set, 
        callbacks=momentum_metrics_history)

hist = momentum_metrics_history[1].monitor_values['accuracies']
for k, v in hist.items():
    if len(v) != 0: plt.plot(v, label=k)
plt.legend()

hists = {
  'Model_0': metrics_history,
  'Model_1': adam_metrics_history,
  'Model_2': momentum_metrics_history
} 

num_epochs = 45

plt.figure(figsize=(15, 3))
plt.subplot(121)
for opt, hist in hists.items():
    h = hist[0].monitor_values['accuracies']['acc']
    h = ut.moving_average(h, periods=100)
    step_to_epoch = np.linspace(0, len(h)/(50000/batch_size), len(h))
    plt.plot(step_to_epoch, h, label=opt, alpha=0.8)
plt.ylim([0, 1])
plt.xlim([0, num_epochs])
plt.legend(fontsize=12, loc='lower right')
plt.xlabel('epoch')
plt.ylabel('training accuracy')

plt.subplot(122)
for opt, hist in hists.items():
    h = hist[0].monitor_values['losses']['loss']
    h = ut.moving_average(h, periods=100)
    step_to_epoch = np.linspace(0, len(h)/(50000/batch_size), len(h))
    plt.plot(step_to_epoch, h, label=opt, alpha=0.8)
plt.legend(fontsize=12, loc='upper right')
plt.xlim([0, num_epochs])
plt.xlabel('epoch')
plt.ylabel('training loss')
# plt.savefig('../img/cifar10_cnn_training.pdf')
# plt.savefig('../img/cifar10_cnn_training.png')

plt.figure(figsize=(15, 3))
plt.subplot(121)

for opt, hist in hists.items():
    h = hist[1].monitor_values['accuracies']['val_acc']
    plt.plot(h[:num_epochs], 'o-',label=opt,  alpha=0.9, linewidth=2)
plt.legend(fontsize=12, loc='lower right')
plt.xlabel('epoch')
plt.ylabel('validation\naccuracy')
plt.xlim([0, num_epochs])

plt.subplot(122)
for opt, hist in hists.items():
    h = hist[1].monitor_values['losses']['val_loss']
    plt.plot(h[:num_epochs], 'o-', label=opt,  alpha=0.9, linewidth=2)
plt.legend(fontsize=12, loc='upper right')
plt.xlabel('epoch')
plt.ylabel('validation\nloss')
plt.xlim([0, num_epochs])
# plt.savefig('../img/cifar10_cnn_validation.pdf')
# plt.savefig('../img/cifar10_cnn_validation.png')