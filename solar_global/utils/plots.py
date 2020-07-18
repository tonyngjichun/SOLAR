import cv2
import numpy as np

import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Arrow, Circle
from mpl_toolkits.axes_grid1 import make_axes_locatable

from solar_global.datasets.datahelpers import default_loader, unnormalise

def plot_ranks(qimages, images, ranks, gnd, bbxs, summary, dataset, epoch=1, n_samples=20, protocol='hard'):

    print('Generating ranking figures... Protocol:', protocol)
    step = 1

    for i in tqdm(range(0, len(qimages), step)):
        g = {}
        if protocol == 'easy':
            g['ok'] = list(gnd[i]['easy'])
            g['junk']= list(gnd[i]['hard']) + list(gnd[i]['junk'])
        elif protocol == 'medium':
            g['ok'] = list(gnd[i]['easy']) +  list(gnd[i]['hard'])
            g['junk']= list(gnd[i]['junk'])
        elif protocol == 'hard':
            g['ok'] = list(gnd[i]['hard'])
            g['junk']= list(gnd[i]['easy']) + list(gnd[i]['junk'])

        n = min(len(g['ok']), n_samples) + 1

        fig = plt.figure(figsize=(10, 10), dpi=200) 
        rows = int(np.floor(np.sqrt(n + 1)))
        cols = int(np.ceil((n + 1) / rows))
        grid = fig.add_gridspec(rows, cols, wspace=0.1, hspace=0.1)

        j  = 0  
        counter = 1
        while j < n:
            ax_img = fig.add_subplot(grid[j])
            ax_img.set_xticks([])
            ax_img.set_yticks([])
            if j == 0:
                ax_img.imshow(default_loader(qimages[i]))
                ax_img.set_title('query')
                bbx = bbxs[i]

                left, bot = bbx[0], bbx[1]
                b_h, b_w = bbx[3] - bbx[1], bbx[2] - bbx[0]

                for spine in ax_img.spines.values():
                    spine.set_edgecolor('black')
                    spine.set_linewidth(4)

                rect = patches.Rectangle((left, bot), b_w, b_h, fill=False, linewidth=3, edgecolor='y', facecolor='None')
                ax_img.add_patch(rect)

                j += 1
            else:
                db_id = ranks[counter-1][i]
                if db_id in g['junk']:
                    counter += 1
                    continue

                edgecolor = 'red'
                if db_id in g['ok']:
                    edgecolor = 'green'
                ax_img.imshow(default_loader(images[db_id]))
                ax_img.set_title('rank{}'.format(j))

                for spine in ax_img.spines.values():
                    spine.set_edgecolor(edgecolor)
                    spine.set_linewidth(4)
                counter +=1 
                j += 1

        fig.suptitle(str(epoch) + dataset + '-' + protocol, fontsize=12)
        summary.add_figure('/' + dataset + '/' + protocol + '/' + str(epoch), fig, global_step=i+1)


def plot_embeddings(images, vecs, summary, imsize=64, sample_freq=1):
    print("Creating embedding visualisation")
    transform = transforms.Compose([
                                transforms.Resize(size=imsize),
                                transforms.CenterCrop(size=imsize),
                                transforms.ToTensor(),
                            ])

    vecs = torch.from_numpy(vecs).permute(1,0)
    _ids = list(range(0, len(images), sample_freq))
    vecs = vecs[_ids]
    label_img = [] 

    for _id in tqdm(_ids):
        label_img.append(transform(default_loader(images[_id])))

    label_img = torch.stack(label_img)

    summary.add_embedding(vecs, label_img=label_img, tag='database')

    


def draw_soa_map(img, model_retr, refPt):

    normalize = transforms.Normalize(mean=model_retr.meta['mean'], std=model_retr.meta['std'])
    transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    img_tensor = transform(img).unsqueeze(0).cuda()
    p = model_retr.pool.p

    fig = plt.figure(dpi=200)
    ax = fig.add_subplot(111)
    # ax2 = fig.add_subplot(121)
    ax.set_xticks([])
    ax.set_yticks([])
    # ax2.set_xticks([])
    # ax2.set_yticks([])

    h, w = img_tensor.shape[-2:]

    with torch.no_grad():
        v_temp, _, soa_m1 = model_retr.features(img_tensor)
        
        h_m1, w_m1 = v_temp.shape[-2:]

        pixel_seeing = torch.zeros(h_m1, w_m1)

        N_m1 = soa_m1.shape[-1]

        # pix_seeing_m1_array = torch.zeros(N_m1)

        pos_hm1, pos_wm1 = int(((refPt[0][1] + refPt[1][1]) / 2) // 32), int(((refPt[0][0] + refPt[1][0]) / 2) // 32)
        pos_h1, pos_w1 = ((refPt[0][1] + refPt[1][1]) / 2), ((refPt[0][0] + refPt[1][0]) / 2)
        # pixel_seeing[pos_hm1, pos_wm1] = 1

        soa_m1 = soa_m1.view(1, h_m1, w_m1, -1)
        self_soa_m1 = soa_m1[:, pos_hm1, pos_wm1, ...].view(-1, h_m1, w_m1)
        self_soa_m1 = F.interpolate(self_soa_m1.unsqueeze(1).cpu(), size=(h, w), mode='bilinear').squeeze()

        # pixel_seeing = F.interpolate(pixel_seeing.unsqueeze(0).unsqueeze(0).cpu(), size=(h, w), mode='bilinear').squeeze()

        ax.imshow(img)
        ax.imshow(self_soa_m1.numpy(), cmap='jet', alpha=.65)

        ax.add_patch(Circle((pos_w1, pos_h1), radius=5, color='white', edgecolor='white', linewidth=5))


        # ax2.imshow(img)
        # ax2.imshow(pixel_seeing.numpy(), cmap='gray', alpha=.85)

        plt.tight_layout()

    # redraw the canvas
    fig.canvas.draw()

    # convert canvas to image
    img_cv2 = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img_cv2  = img_cv2.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # img_cv2 is rgb, convert to opencv's default bgr
    img_cv2 = cv2.cvtColor(img_cv2,cv2.COLOR_RGB2BGR)
    
    return img_cv2