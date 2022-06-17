from __future__ import division
from __future__ import print_function
import os, glob, shutil, math, json
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2

def get_gauss_kernel(size, sigma):
    '''Function to mimic the 'fspecial' gaussian MATLAB function'''
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()


def batchGray2Colormap(gray_batch):
    colormap = plt.get_cmap('viridis')
    heatmap_batch = []
    for i in range(gray_batch.shape[0]):
        # quantize [-1,1] to {0,1}
        gray_map = gray_batch[i, :, :, 0]
        heatmap = (colormap(gray_map) * 2**16).astype(np.uint16)[:,:,:3]
        heatmap_batch.append(heatmap/127.5-1.0)
    return np.array(heatmap_batch)


def visualizeLossCurves(data_dir, epoch=2):
    #! when only one sample in the file, error will occur in "len(data)" below
    if epoch < 2:
        return
    file_list = []
    for file_path in glob.glob(os.path.join(data_dir, '*')):
        if file_path.endswith('.png') or os.path.isdir(file_path):
            continue
        file_list.append(file_path)
    if (len(file_list) == 0):
        return
    fig_ncol = math.ceil(len(file_list) / 2.)
    #print('file num:', len(file_list))
    plt.figure(figsize=(fig_ncol*6, 10), dpi = 100*2, tight_layout=True)
    #plt.subplots_adjust(wspace=0.3, hspace=0.3)
    for i in range(2):
        for j in range(fig_ncol):
            no = i*fig_ncol+j
            if no >= len(file_list):
                break
            file_path = file_list[no]
            _, file_name = os.path.split(file_path)
            data = np.loadtxt(file_path)
            x = np.linspace(1,len(data),len(data))
            plt.sca(plt.subplot(2,fig_ncol,no+1))
            plt.title(file_name)
            plt.plot(x, data, color='red', linewidth=1)
    plt.show()
    plt.savefig(os.path.join(data_dir,'loss_plot.png'))
    plt.close()


def save_images_from_batch(img_batch, save_dir, filename_list, batch_no=-1):
    N,H,W,C = img_batch.shape
    if C == 3:
        #! rgb color image
        for i in range(N):
            # [-1,1] >>> [0,255]
            img_batch_i = np.clip(img_batch[i,:,:,:]*0.5+0.5, 0, 1)
            image = Image.fromarray((255.0*img_batch_i).astype(np.uint8))
            save_name = filename_list[i] if batch_no==-1 else '%05d.png' % (batch_no*N+i)
            image.save(os.path.join(save_dir, save_name), 'PNG')
    elif C == 1:
        #! single-channel gray image
        for i in range(N):
            # [-1,1] >>> [0,255]
            img_batch_i = np.clip(img_batch[i,:,:,0]*0.5+0.5, 0, 1)
            image = Image.fromarray((255.0*img_batch_i).astype(np.uint8))
            save_name = filename_list[i] if batch_no==-1 else '%05d.png' % (batch_no*img_batch.shape[0]+i)
            image.save(os.path.join(save_dir, save_name), 'PNG')
    else:
        #! multi-channel: save each channel as a single image
        for i in range(N):
            # [-1,1] >>> [0,255]
            for j in range(C):
                img_batch_ij = np.clip(img_batch[i,:,:,j]*0.5+0.5, 0, 1)
                image = Image.fromarray((255.0*img_batch_ij).astype(np.uint8))
                if batch_no == -1:
                    _, file_name = os.path.split(filename_list[i])
                    name_only, _ = os.path.os.path.splitext(file_name)
                    save_name = name_only + '_c%d.png' % j
                else:
                    save_name = '%05d_c%d.png' % (batch_no*N+i, j)
                image.save(os.path.join(save_dir, save_name), 'PNG')
    return None


def save_normLabs_from_batch(img_batch, save_dir, filename_list, batch_no=-1):
    N,H,W,C = img_batch.shape
    if C != 3:
        print('@Warning:the Lab images are NOT in 3 channels!')
        return None
    # denormalization: L: (L+1.0)*50.0 | a: a*110.0| b: b*110.0
    img_batch[:,:,:,0] = img_batch[:,:,:,0] * 50.0 + 50.0
    img_batch[:,:,:,1:3] = img_batch[:,:,:,1:3] * 110.0
    #! convert into RGB color image
    for i in range(N):
        rgb_img = cv2.cvtColor(img_batch[i,:,:,:], cv2.COLOR_LAB2RGB)
        image = Image.fromarray((rgb_img*255.0).astype(np.uint8))
        save_name = filename_list[i] if batch_no==-1 else '%05d.png' % (batch_no*N+i)
        image.save(os.path.join(save_dir, save_name), 'PNG')
    return None


def get_filelist(data_dir):
    file_list = glob.glob(os.path.join(data_dir, '*.*'))
    file_list.sort()
    return file_list
    

def collect_filenames(data_dir):
    file_list = get_filelist(data_dir)
    name_list = []
    for file_path in file_list:
        _, file_name = os.path.split(file_path)
        name_list.append(file_name)
    name_list.sort()
    return name_list


def exists_or_mkdir(path, need_remove=False):
    if not os.path.exists(path):
        os.makedirs(path)
    elif need_remove:
        shutil.rmtree(path)
        os.makedirs(path)
    return None


def save_list(save_path, data_list, append_mode=False):
    n = len(data_list)
    if append_mode:
        with open(save_path, 'a') as f:
            f.writelines([str(data_list[i]) + '\n' for i in range(n-1,n)])
    else:
        with open(save_path, 'w') as f:
            f.writelines([str(data_list[i]) + '\n' for i in range(n)])
    return None
    
    
def save_dict(save_path, dict):
    json.dumps(dict, open(save_path,"w"))
    return None


if __name__ == '__main__':
    data_dir = '../PolyNet/PolyNet/cache/'
    print('Hello, world!')