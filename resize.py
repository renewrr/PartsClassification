from PIL import Image
import os, sys

path = "./PartSort/img/"
outpath = "./PartSort/img/resize/"
dirs = os.listdir( path )
final_size = 100;

def resize_aspect_fit():
    for item in dirs:
         if item == '.DS_Store':
             print(item)
             continue
         if os.path.isfile(path+item):
             im = Image.open(path+item)
             f, e = os.path.splitext(path+item)
             size = im.size
             ratio = float(final_size) / max(size)
             temp = [int(x*ratio) for x in size]
             if(temp[0] <= 0): temp[0] = 1
             if(temp[1] <= 0): temp[1] = 1
             new_image_size = tuple(temp)
             im = im.resize(new_image_size, Image.ANTIALIAS)
             new_im = Image.new("RGB", (final_size, final_size),color = tuple([255,255,255]))
             new_im.paste(im, ((final_size-new_image_size[0])//2, (final_size-new_image_size[1])//2))
             new_im.save(outpath+item, quality=90)
resize_aspect_fit()