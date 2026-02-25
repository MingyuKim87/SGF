import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
import os
from moviepy.editor import *

def animate_imlist(im_list,anim_name="movie"):
    """
    Takes in list of image paths.
    outputs animation
    """
    fig, ax = plt.subplots()

    ims = []
    for p in im_list:
        im  = plt.imshow(p)
        ims.append(im)
    import ipdb; ipdb.set_trace()
    ani = animation.ArtistAnimation(fig, ims, interval=10, blit=True,
                                    repeat_delay=1000)

    ani.save(f"{anim_name}.mp4")


def animate(im_list,anim_name="movie"):
    clips = [ImageClip(m).set_duration(2)
        for m in im_list]

    concat_clip = concatenate_videoclips(clips, method="compose")
    concat_clip.write_videofile(f"animations/{anim_name}.mp4", fps=30)


def get_paths_with_keys(keyword,imagespath):
    filenames = []
    for file in os.listdir(imagespath):
        if keyword in file:
            filenames.append(os.path.join(imagespath, file))
    filenames.sort()
    return filenames

def animation_kws(keyword,imagespath,animation_name="movie"):
    fnames = get_paths_with_keys(keyword,imagespath)
    # import ipdb; ipdb.set_trace()
    animate(fnames,anim_name=animation_name)


classn = "church"
clevel = "classlevel"
nit = "30"

animation_kws(classn,f"./output_imagenette2class/run_cosine_{nit}k_bs16_classlevel/generations",animation_name=f"2class_{classn}_{clevel}_{nit}k_anim")

