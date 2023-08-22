from imutils import paths
import argparse
import os
from PIL import Image
import numpy as np
from scipy.misc import imsave
from LadderNetv65 import *
import cv2
from natsort import natsorted
#import configparser
from help_functions import group_images
from pre_processing import my_PreProc
import sys
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader,Dataset
sys.path.insert(0, '../')
from help_functions import *
from extract_patches import recompone
from extract_patches import recompone_overlap
from extract_patches import paint_border_overlap
from extract_patches import extract_ordered_overlap
import pdb
ap=argparse.ArgumentParser()
ap.add_argument("--input","--arg1",type=str,default="./input/")
ap.add_argument("--result","--arg2",type=str,default="./result_a/")
args=vars(ap.parse_args())

dir_input = "./input/"
dir_result = "./result_a/"
if not os.path.exists(dir_input):
    os.makedirs(dir_input)
    
if not os.path.exists(dir_result):
    os.makedirs(dir_result)


folder_input=args["input"]
folder_result=args["result"]

if not os.path.exists(folder_input):
    os.makedirs(folder_input)
    
if not os.path.exists(folder_result):
    os.makedirs(folder_result)

dataPaths=sorted(list(paths.list_images(folder_input)))
labelPaths=sorted(list(paths.list_images(folder_result)))


print("input images in:"+folder_input)
print("result of prediction in:"+folder_result)


dem=0
choose_rate=133
h_img=588
w_img=568
h_win=588
w_win=568
patch_smallest_height=48
patch_smallest_width=48
average_mode = True
N_visual=1


path_experiment=folder_result

channels_patch = 3

stride_height=5
stride_width=5

   
dataPaths=natsorted(list(paths.list_images(folder_input)))  
test_lst=[dataPath.strip()for dataPath in dataPaths]
tam=[]
dem=0


check_path2 = 'LadderNetv65_layer_%d_filter_%d.pt7'% (4,10) #'UNet16.pt7'#'UNet_Resnet101.pt7'
    
net2 = LadderNetv6(num_classes=2,layers=4,filters=10,inplanes=1)

device2 = 'cuda' if torch.cuda.is_available() else 'cpu'

resume = True

if device2 == 'cuda':
    net2.cuda()
    net2 = torch.nn.DataParallel(net2, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

if resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/'+check_path2)
    net2.load_state_dict(checkpoint['net'])
    start_epoch = checkpoint['epoch']    
  

for idx in range(0, len(test_lst)): 
    im = Image.open(test_lst[idx])

    
    
    pos = [i for i, ltr in enumerate(test_lst[idx]) if ltr == '/']
    print (test_lst[idx][pos[-1]+1:])
    
    
    in_ = np.array(im, dtype=np.float32)
    

    
    height = in_.shape[0]
    width = in_.shape[1]
    #pdb.set_trace()
    
    rate=int((np.minimum(height,width)-48)/5)

    if(rate>=choose_rate):
        rate=choose_rate     
    
    h_img=rate*5+48
    w_img=h_img
    h_win=h_img
    w_win=h_img
    
    
    output_conv3_2 = np.zeros((in_.shape[0], in_.shape[1]), dtype=np.float32)
    
    #pdb.set_trace()
    patch_2=np.zeros((h_win, w_win, 3), dtype=np.uint8)
    rgb_2 = np.zeros((h_win, w_win, 3), dtype=np.uint8)


                      
    if ((w_win <= width) and (h_win <= height)):
        print (test_lst[idx][pos[-1]+1:])            
        for r in range(0, height, h_win):
            for c in range(0, width, w_win):
                b_b = r+h_win
                r_b = c+w_win              
       
        
                if (b_b >= height):
                    b_b=height 
                    r=b_b - h_win                             
                if (r_b >= width):
                    r_b = width 
                    c = r_b - w_win

                if(len(in_.shape)==2):
                    patch_2=np.zeros((h_img, w_img), dtype=np.uint8)
                    patch_2 = in_[r:b_b, c:r_b]

                    #rgb_mask=255*np.ones((h_win, w_win, 3), dtype=np.uint8)
                    rgb_2[:,:,0]=patch_2
                    rgb_2[:,:,1]=patch_2
                    rgb_2[:,:,2]=patch_2               
               
                
                if(len(in_.shape)==3):
                    patch_2=np.zeros((h_img, w_img, 3), dtype=np.uint8)
                    patch_2[:,:,0] = in_[r:b_b, c:r_b,0]
                    patch_2[:,:,1] = in_[r:b_b, c:r_b,1]
                    patch_2[:,:,2] = in_[r:b_b, c:r_b,2]
                   
                    rgb_2=patch_2

                    
                    
                imgs = np.empty((1,h_img,w_img,channels_patch))
               
                imgs[0] = rgb_2
                
                imgs = np.transpose(imgs,(0,3,1,2))
                assert(imgs.shape == (1,channels_patch,h_img,w_img))
                #pdb.set_trace()
                test_imgs = my_PreProc(imgs)
                test_imgs = paint_border_overlap(test_imgs, patch_smallest_height, patch_smallest_width, stride_height, stride_width)
                #pdb.set_trace()
                patches_imgs_test = extract_ordered_overlap(test_imgs,patch_smallest_height,patch_smallest_width,stride_height,stride_width)
                
                new_height=test_imgs.shape[2]
                new_width=test_imgs.shape[3]                   
                
                
              
                class TrainDataset(Dataset):
                    """Endovis 2018 dataset."""
                
                    def __init__(self, patches_imgs):
                        self.imgs = patches_imgs
                
                    def __len__(self):
                        return self.imgs.shape[0]
                
                    def __getitem__(self, idx):
                        return torch.from_numpy(self.imgs[idx,...]).float()
                
                batch_size = 512
                
                test_set = TrainDataset(patches_imgs_test)
                test_loader = DataLoader(test_set, batch_size=batch_size,
                                          shuffle=False, num_workers=4)
                
                #pdb.set_trace()
                
                preds = []
                for batch_idx, inputs in enumerate((test_loader)):
                    
                    #pdb.set_trace()
                    
                    inputs = inputs.to(device2)
                    
                    outputs = net2(inputs)
           
                    
                    outputs = torch.nn.functional.softmax(outputs,dim=1)
                    outputs = outputs.permute(0,2,3,1)
                   
                    
                    shape = list(outputs.shape)
                    
                    
                    
                    outputs = outputs.view(-1,shape[1]*shape[2],2)
                    
                    outputs = outputs.data.cpu().numpy() 
                    
                    preds.append(outputs)
                    
                    
                
                predictions = np.concatenate(preds,axis=0)
                print("Predictions 1 patch finished")
                #===== Convert the prediction arrays in corresponding images
                pred_patches = pred_to_imgs(predictions, patch_smallest_height, patch_smallest_width,"original")
                

                #========== Elaborate and visualize the predicted images ====================
                pred_imgs = None
                
                if average_mode == True:
                    pred_imgs = recompone_overlap(pred_patches, new_height, new_width, stride_height, stride_width)# predictions
                
                
                else:
                    pred_imgs = recompone(pred_patches,13,12)       # predictions
                
                

                
                pred_imgs = pred_imgs[:,:,0:h_img,0:w_img]
                
                
                #print("pred imgs shape: " +str(pred_imgs.shape))
                

                tam=chuyen_format_anh(group_images(pred_imgs,N_visual))                  
                

                tmp = output_conv3_2[r:b_b, c:r_b]
                #tmp = np.maximum(tmp, im_thresh0)
                tmp = np.maximum(tmp,tam)
                output_conv3_2[r:b_b, c:r_b] = tmp
                final = output_conv3_2
                pos = [i for i, ltr in enumerate(test_lst[idx]) if ltr == '/']
                print (test_lst[idx][pos[-1]+1:])
                
                ret,im_thresh1 = cv2.threshold(final,170,255,cv2.THRESH_BINARY)                    
                
                imsave(path_experiment + test_lst[idx][pos[-1]+1:],255-im_thresh1)
                    

