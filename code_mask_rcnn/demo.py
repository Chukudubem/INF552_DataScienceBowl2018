# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 21:46:35 2018

@author: chaofan
"""

import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import skimage
os.chdir(r'C:\Users\chaofan\Desktop\2018 spring term\INF 552\project\code\Mask_RCNN-master')
from config import Config
import utils
import model as modellib
import visualize
from model import log
# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)
    

class ShapesConfig(Config):

    # Give the configuration a recognizable name
    NAME = "nucleis"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1 # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 128

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 2

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5
    
config = ShapesConfig()
config.display()




def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

    
###############################
# kaggle data 
##############################
class NucleiDataset(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """

    def load_nucleis(self,id_list,path):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        self.add_class("nucleis", 1, "nuclei")
 
        # Add images
        
        for image_id in id_list:
            image_path = os.path.join(path, image_id, 'images', '{}.png'.format(image_id))
            mask_len =len( os.listdir(os.path.join(path, image_id, 'masks')))
      
            
            self.add_image(source= "nucleis", image_id=image_id, path=image_path,nucleis=["nuclei"]*mask_len  )

   
    def load_image(self, count):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but
        in this case it generates the image on the fly from the
        specs in image_info.
        """
        info = self.image_info[count]
           # bg_color = np.array(info['bg_color']).reshape([1, 1, 3])
        image_path= info['path']
        image = skimage.io.imread(image_path)[:,:,:3]  # rgb 
  
        return image
	
    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "nucleis":
            return info["nucleis"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        path = 'D:\\Kaggle\\data\\stage1_train'
        info = self.image_info[image_id]
        image_id = info['id']
        nucleis = info['nucleis']
        #count = len(nucleis)
        
        
        #mask = np.zeros([info['height'], info['width'], count], dtype=np.uint8)
        file_list=[os.path.join(path, image_id, 'masks',x) for x in  os.listdir(os.path.join(path, image_id, 'masks'))]
        mask_paths =file_list
        

        masks = skimage.io.imread_collection(mask_paths).concatenate()

        masks = np.moveaxis(masks, 0, -1)
        

        # Handle occlusions
      #  occlusion = np.logical_not(masks[:, :, -1]).astype(np.uint8)
       # for i in range(count-2, -1, -1):
        #    masks[:, :, i] = masks[:, :, i] * occlusion
         #   occlusion = np.logical_and(occlusion, np.logical_not(masks[:, :, i]))
        # Map class names to class IDs.
        class_ids = np.array([self.class_names.index(s) for s in nucleis])
        return masks, class_ids.astype(np.int32)

    
# Training dataset
dataset_train = NucleiDataset()

path= r'D:\Kaggle\data\stage1_train'
id_list=os.listdir(path)
#len(id_list) 670
dataset_train.load_nucleis(id_list[:500],path)
dataset_train.prepare()

# Validation dataset
dataset_val = NucleiDataset()
dataset_val.load_nucleis(id_list[-20:],path)
dataset_val.prepare()
    
# test data 
test_path = r'D:\Kaggle\data\stage1_test'
test_id_list=os.listdir(test_path)
#dataset_test = NucleiDataset()
#dataset_test.load_nucleis(test_id_list,test_path)
#dataset_test.prepare()
    




# Load and display random samples
image_ids = np.random.choice(dataset_train.image_ids, 4)
for count in image_ids:
    
    image = dataset_train.load_image(count)
   # image_id = dataset_train.image_info[count]
    mask, class_ids = dataset_train.load_mask(count)
    visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)
    
    
    
    
    
# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

# Which weights to start with?
init_with = "coco"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last()[1], by_name=True)
    

'''
Train in two stages:

Only the heads. Here we're freezing all the backbone layers and training 
only the randomly initialized layers (i.e. the ones that we didn't use 
pre-trained weights from MS COCO). To train only the head layers, pass 
layers='heads' to the train() function.

Fine-tune all layers. For this simple example it's not necessary, but we're 
including it to show the process. Simply pass layers="all to train all layers.
'''




# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.’
'''
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE, 
            epochs=1, 
            layers='heads')
''' 
# Fine tune all layers
# Passing layers="all" trains all layers. You can also 
# pass a regular expression to select which layers to
# train by name pattern.
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE,
            epochs=2, 
            layers="all")


# Save weights
# Typically not needed because callbacks save after every epoch
# Uncomment to save manually
# model_path = os.path.join(MODEL_DIR, "mask_rcnn_shapes.h5")
# model.keras_model.save_weights(model_path)


class InferenceConfig(ShapesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
model_path1 =r'C:\Users\chaofan\Desktop\change1.h5' # model.find_last()[1]
model_path2 =r'C:\Users\chaofan\Desktop\change2.h5' # model.find_last()[1]
model_path3 =r'C:\Users\chaofan\Desktop\change3.h5' # model.find_last()[1]
model_path4 =r'C:\Users\chaofan\Desktop\change4.h5' # model.find_last()[1]
model_path5 = r'C:\Users\chaofan\Desktop\mask_rcnn_nucleus_0010.h5'
# Load trained weights (fill in path to trained weights here) 
for model_path in [model_path1,model_path2,model_path3,model_path4]:
    assert model_path != "", "Provide path to trained weights"
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)
    
    # Test on a random image
    
    image_id = 14#random.choice(dataset_val.image_ids)
    
    original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset_val, inference_config, 
                               image_id, use_mini_mask=False)
    
    
    
    #log("original_image", original_image)
    #log("image_meta", image_meta)
    #log("gt_class_id", gt_class_id)
    #log("gt_bbox", gt_bbox)
    #log("gt_mask", gt_mask)
    
    #visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id, 
    #                            dataset_train.class_names, figsize=(8, 8))
    
    
    
    
    
    results = model.detect([original_image], verbose=1)
    
    
    r = results[0]
    visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], 
                                dataset_val.class_names, r['scores'], ax=get_ax())
    
    
    
    
    # Compute VOC-Style mAP @ IoU=0.5
    # Running on 50 images. Increase for better accuracy.
    image_ids = np.random.choice(dataset_val.image_ids,20)
    APs = []
    for image_id in image_ids:
        # Load image and ground truth data
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(dataset_val, inference_config,
                                   image_id, use_mini_mask=False)
        molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
        # Run object detection
        results = model.detect([image], verbose=0)
        r = results[0]
        # Compute AP
        if len(r['masks']) != 0:
            AP, precisions, recalls, overlaps =\
                utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                                 r["rois"], r["class_ids"], r["scores"], r['masks'])
            APs.append(AP)
        
    print("mAP: ", np.mean(APs))
    


######################
# test data 
#######################
import pandas as pd


from skimage.morphology import label # label regions
def rle_encoding(x):
    '''
    x: numpy array of shape (height, width), 1 - mask, 0 - background
    Returns run length as list
    '''
    dots = np.where(x.T.flatten()==1)[0] # .T sets Fortran order down-then-right
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b+1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def prob_to_rles(x, cut_off = 0.5):
    lab_img = label(x>cut_off)
    if lab_img.max()<1:
        lab_img[0,0] = 1 # ensure at least one prediction per image
    for i in range(1, lab_img.max()+1):
        yield rle_encoding(lab_img==i)


def get_rles(mask_image):
    #li=[]
  #  for i in range(np.shape(r['masks'])[-1]):
  #      
   #     li.append([r['masks'][:,:,i]])
        
    df = pd.DataFrame(mask_image)
    df.columns= ['masks']
    
    rles = df['masks'].map(lambda x: list(prob_to_rles(x)))

    return rles

#for _, c_row in test_img_df.iterrows():


out_pred_list = []
class_names= ['BG','nuclei']
os.chdir(r'D:\Kaggle\data\stage_1_test_results')
for image_id in test_id_list:
    print(image_id)
    image_path = os.path.join(test_path, image_id, 'images', '{}.png'.format(image_id))
    image = skimage.io.imread(image_path)[:,:,:3]

    # Run detection
    results = model.detect([image], verbose=1)

    # Visualize results
    r = results[0]
    mask_image = r['masks'].sum(axis = 2)
    mask_image[mask_image > 1] =1
    rles = list(prob_to_rles(mask_image))
    #visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
     #                           class_names, r['scores'],ax=get_ax())
    
   # visualize.save_instances('predict_'+image_id,image, r['rois'], r['masks'], r['class_ids'], 
    #                            class_names, r['scores'])
    


    for c_rle in rles:
        out_pred_list+=[dict(ImageId=image_id, 
                             EncodedPixels = ' '.join(np.array(c_rle).astype(str)))]
    
out_pred_df = pd.DataFrame(out_pred_list)  

cols =[ 'ImageId','EncodedPixels']

out_pred_df = out_pred_df[cols]

out_pred_df.to_csv(r'D:\Kaggle\data\submission.csv ',index = None )
