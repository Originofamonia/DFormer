import os
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import pathlib
import matplotlib as mpl
from pptx import Presentation
from pptx.util import Inches, Pt
import pandas as pd
from pathlib import Path

import numpy as np


def compare_inferred_masks():
    model_paths = ['epoch-2_miou_85', 'epoch-3_miou_92', 'epoch-4_miou_94', 'epoch-5_miou_95']
    colors = ['#00000000', 'lime']
    cmap = ListedColormap(colors)
    prs = Presentation()
    blank_slide_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank_slide_layout)
    left = top = Inches(0.1)
    top_2 = Inches(6)
    width = Inches(14.0)
    height = Inches(1.2)
    alpha = 0.6
    df = pd.read_csv(f'datasets/trav/df2.csv', index_col=0)
    for row in df.itertuples(index=True, name='Pandas'):
        img_id = row.img.split('/')[-1].strip('.jpg')
        gt_path = row.img.replace('/images/', '/labels/')
        gt_file = os.path.splitext(gt_path)[0] + '.npy'
        ep_2_filename = f'output/{model_paths[0]}/{img_id}.npy'
        ep_3_filename = f'output/{model_paths[1]}/{img_id}.npy'
        ep_4_filename = f'output/{model_paths[2]}/{img_id}.npy'
        ep_5_filename = f'output/{model_paths[3]}/{img_id}.npy'
        slide = prs.slides.add_slide(blank_slide_layout)
        fig, axs = plt.subplots(2, 3, figsize=(14, 6))  # w,h
        # img, epoch2, epoch3
        #  gt, epoch4, epoch5
        q_img = plt.imread(row.img)
        q_target = np.load(gt_file)
        ep2 = np.load(ep_2_filename)
        ep3 = np.load(ep_3_filename)
        ep4 = np.load(ep_4_filename)
        ep5 = np.load(ep_5_filename)
        axs[0,0].imshow(q_img)
        axs[0,0].set_title(f'img')
        axs[0,0].axis('off')

        axs[1,0].imshow(q_img)
        axs[1,0].imshow(q_target, cmap=cmap, alpha=alpha)
        axs[1,0].set_title(f'target')
        axs[1,0].axis('off')

        axs[0,1].imshow(q_img)
        axs[0,1].imshow(ep2, cmap=cmap, alpha=alpha)
        axs[0,1].set_title(f'ep2')
        axs[0,1].axis('off')

        axs[0,2].imshow(q_img)
        axs[0,2].imshow(ep3, cmap=cmap, alpha=alpha)
        axs[0,2].set_title(f'ep3')
        axs[0,2].axis('off')

        axs[1,1].imshow(q_img)
        axs[1,1].imshow(ep4, cmap=cmap, alpha=alpha)
        axs[1,1].set_title(f'ep4')
        axs[1,1].axis('off')

        axs[1,2].imshow(q_img)
        axs[1,2].imshow(ep5, cmap=cmap, alpha=alpha)
        axs[1,2].set_title(f'ep5')
        axs[1,2].axis('off')

        plt.subplots_adjust(hspace=0.01, wspace=0.01)
        img_filename = f'output/pptx/{img_id}.png'
        fig.savefig(img_filename, bbox_inches='tight', pad_inches=0)
        plt.close()
        pic = slide.shapes.add_picture(img_filename, left, top)
        text_box = slide.shapes.add_textbox(left, top_2, width, height)

        # Get the text frame within the text box
        tf = text_box.text_frame

        # Add a paragraph to the text frame
        p = tf.add_paragraph()
        p.text = f'img: {row.img}'
    
    prs.save(f'output/pptx/different_models.pptx')


def draw_few_shot():
    """
    TODO: random 1 row from df1 as query and random 1 row from df2 as support
    """
    pass

if __name__ == '__main__':
    compare_inferred_masks()
