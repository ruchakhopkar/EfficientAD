from anomalib.data.folder import Folder
from anomalib.data.task_type import TaskType
from pathlib import Path
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from anomalib.data import InferenceDataset
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
from os import walk
import os 
import pandas as pd
from PIL import Image
import numpy as np
from anomalib.models import Padim
from anomalib.post_processing import NormalizationMethod, ThresholdMethod
from anomalib.utils.callbacks import (
        MetricsConfigurationCallback,
        MinMaxNormalizationCallback,
        PostProcessingConfigurationCallback,
        ImageVisualizerCallback,
        MetricVisualizerCallback,
        )


dataset_name = 'XNPS'
train_test_split = 0.2
image_size = 256
seed = 42
max_epochs = 1
train = False
path_to_ckpt = '/home/ruchak/using_efficientAD/tb_logs/my_model/version_1/checkpoints/epoch=0-step=11.ckpt'

#create the datamodule
datamodule = Folder(root = Path.cwd() / dataset_name,
        normal_dir = "normal",
        abnormal_dir = "abnormal",
        normal_split_ratio = train_test_split,
        image_size = (image_size, image_size),
        task = TaskType.CLASSIFICATION,
        seed = 42)

datamodule.setup() #split data into train, val, test datasets
datamodule.prepare_data() #create dataloaders

i, data = next(enumerate(datamodule.val_dataloader()))

#we will now load the model
model=Padim(
        input_size = (image_size, image_size),
        backbone = 'resnet18',
        layers = ['layer1', 'layer2', 'layer3'],
        )

#logger
logger = TensorBoardLogger('tb_logs', name = 'my_model')

#these are certain callbacks that are all available in the anomalib library 
#the callbacks are used to make certain decisions while training


callbacks = [
        MetricsConfigurationCallback(
            task = TaskType.CLASSIFICATION,
            image_metrics=['AUROC'], #we are maximizing on the AUROC, i.e, we are training the model to always improve the AUROC
        ),
        ModelCheckpoint(
            mode='max',
            monitor='image_AUROC', #we will always save the model that gives us the best AUROC
        ),
        PostProcessingConfigurationCallback(
            normalization_method = NormalizationMethod.MIN_MAX,
            threshold_method = ThresholdMethod.ADAPTIVE,     #after postprocessing, we will normalize the data
        ),
        MinMaxNormalizationCallback(),
        ImageVisualizerCallback(mode = 'full', task = TaskType.CLASSIFICATION, image_save_path = './results/images'),
        MetricVisualizerCallback(mode = 'full', task = TaskType.CLASSIFICATION, image_save_path = './results/images'), #the results on training data will be stored in results/images
        EarlyStopping(monitor = 'image_AUROC', mode = 'max', patience = 3),
        LearningRateMonitor(logging_interval = 'step')
        
]

#this is the trainer that will train the model
trainer = Trainer(
        callbacks = callbacks,
        accelerator = 'cuda' if torch.cuda.is_available() else 'cpu', #we have set the accelerator as cuda. If cuda is not available on the system, we can set it to cpu as well
        devices = 1, #sometime we can have more than 1 GPUs on a system. In that case, we can change this to the number of different GPUs we have
        auto_scale_batch_size = False,
        check_val_every_n_epoch = 1, ##we will validate on a part of the training data as we go. This tells after how many epochs, do we want to evaluate after
        max_epochs = max_epochs, #maximum epochs
        num_sanity_val_steps = 0,
        val_check_interval = 1.0,
        logger = logger
        )

if train:
    #this is where the training happens
    trainer.fit(model = model, datamodule = datamodule)
else:
    model = Padim.load_from_checkpoint(path_to_ckpt)

#testing on the test dataset
test_results = trainer.test(model = model, datamodule = datamodule)

#save the testing results
f = []
for (dirpath, dirnames, filenames) in walk(Path.cwd() / dataset_name / 'test/'):
    for name in filenames:
        f.append(os.path.join(dirpath, name))

df = pd.DataFrame()
anomaly_masks, input_path, pred_score, pred_labels, mask_path, box_path = [], [], [], [], [], []
for i in range(len(f)):
    #make the dataset and the dataloader
    inference_dataset = InferenceDataset(path = f[i], image_size = (image_size, image_size))
    inference_dataloader = DataLoader(dataset = inference_dataset)
    
    #get the predictions from the model
    predictions = trainer.predict(model = model, dataloaders = inference_dataloader)[0]
    
    #store all the results in a dataframe
    input_path.append(predictions['image_path'][0])
    pred_score.append(predictions['pred_scores'].numpy()[0])
    pred_labels.append(int(predictions['pred_labels'].numpy()[0]))
    #save the anomaly maps
    os.makedirs(Path.cwd() / 'results' / dataset_name / 'anomaly_maps', mode = 0o777, exist_ok = True)
    image_name = f[i].split('/')[-1]
    im = Image.fromarray((predictions['anomaly_maps'].numpy()[0, 0] * 255).astype(np.uint8))
    im.save(Path.cwd() / 'results' / dataset_name / 'anomaly_maps' / image_name)
    anomaly_masks.append(Path.cwd() / 'results' / dataset_name / 'anomaly_maps' / image_name)
    #save the predicted masks
    im = Image.fromarray((predictions['pred_masks'].numpy()[0,0]*255).astype(np.uint8))
    os.makedirs(Path.cwd() / 'results' / dataset_name / 'pred_masks' , mode = 0o777, exist_ok = True)
    im.save(Path.cwd() / 'results' / dataset_name / 'pred_masks' / image_name)
    mask_path.append(Path.cwd() / 'results' / dataset_name / 'pred_masks' / image_name)
    box_path.append(predictions['pred_boxes'][0])

#save the dataframe
df['Image Path'] = input_path
df['Pred Score'] = pred_score
df['Pred Labels'] = pred_labels
df['Anomaly Masks Path'] = anomaly_masks
df['Mask Path'] = mask_path
df['Box'] = box_path
df.to_csv(Path.cwd() / 'results' / dataset_name / 'results.csv', index = False)

