# from PIL import ImageSequence
from PIL import Image
# import gif
# from images2gif import writeGif
# from PIL import Image
# import moviepy.editor as mpy
import numpy as np
import os
import scipy.misc
from scipy.misc import imsave, imresize
import imageio
import numpy as np
# import fcn_utils
from utils import *
import tensorflow as tf
from skimage.draw import *

GIFDURATION = 0.2


def dump2disk(vis_dir, step, i1_g,i2_g,i2_warped, depth,blended_image):
    try:
        os.mkdir(vis_dir)
    except:
        pass

    vislst = {}

    getfile = lambda name: os.path.join(vis_dir,name+ "_"+'{:08}'.format(step) +".png")
    getfilegif = lambda name: os.path.join(vis_dir,name+ "_"+'{:08}'.format(step)+".gif")
    savefile = lambda name, img: imsave(getfile(name), img)
    def saveim(name, img):
        if img.ndim == 2:
            h = img.shape[0]
            w = img.shape[1]
            img = np.reshape(img, (h, w, 1))
        if img.shape[2]==1:
            img = np.tile(img,(1,1,3))
        imsave(getfile(name), img[0,:,:,:])

    def savegif(name, img1, img2):
        # imgs = [img1[0,:,:,:],img2[0,:,:,:]]
        img1=(img1)*255.0/np.max(img1)
        img2=(img2)*255.0/np.max(img2)
        imgs = [img1[0,:,:,0],img2[0,:,:,0]]
        imageio.mimsave(getfilegif(name), imgs, 'GIF', duration=GIFDURATION)
        
    def savegif3(name, img1, img2, img3):
        imgs = [img1[0,:,:,:],img2[0,:,:,:],img3[0,:,:,:],img2[0,:,:,:]]
        imageio.mimsave(getfilegif(name), imgs, 'GIF', duration=GIFDURATION)
    def savegif5(name, img1, img2, img3, img4, img5):
        imgs = [img1[0,:,:,:],
                img2[0,:,:,:],
                img3[0,:,:,:],
                img4[0,:,:,:],
                img5[0,:,:,:],
                img4[0,:,:,:],
                img3[0,:,:,:],
                img2[0,:,:,:]]
        imageio.mimsave(getfilegif(name), imgs, 'GIF', duration=GIFDURATION/2)

    ################
    saveim('i1_g', i1_g)
    saveim('i2_g', i2_g)
    saveim('i2w', i2_warped)
    savegif('i1i2', i1_g, i2_g)
    savegif('i1i2w',i2_g, i2_warped)
    saveim('depth', depth)
    saveim('b', blended_image)