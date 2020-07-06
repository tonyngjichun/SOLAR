import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.patches as patches
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

def plot_ranks_and_attentions(model_retr, qimages, images, ranks, gnd, bbxs, summary, dataset, n_samples=20, protocol='hard'):
    def get_attn(model_retr, image):

        image_tensor = transforms.ToTensor()(image).unsqueeze(0).cuda()
        image_tensor = unnormalise(image_tensor)

        h, w = image_tensor.shape[-2:]

        try:
            fmap = model_retr.features(image_tensor)
        except:
            fmap, _, _ = model_retr.features(image_tensor)
            
        p = model_retr.pool.p

        attn = torch.pow(fmap[0].clamp(min=1e-6), p)
        attn = torch.sum(attn, -3, keepdim=True)
        if len(attn.shape) == 3:
            attn = attn.unsqueeze(0)

        attn = F.interpolate(attn, size=(h, w), mode='bilinear')
        attn = (attn - attn.min()) / (attn.max() - attn.min())
        attn = attn.squeeze().detach().cpu()

        return attn

    print('Generating attention figures... Protocol:', protocol)
    step = 1
    if dataset.startswith('megadepth'):
        step = 15
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

        n = min(len(g['ok']), n_samples)# + 1

        print(n)

        # simply plot the images
        fig = plt.figure(dpi=400) 
        grid = fig.add_gridspec(1, n_samples, wspace=0.1, hspace=0.1)

        fig_attn = plt.figure(dpi=400) 
        grid_attn = fig.add_gridspec(1, n_samples, wspace=0.1, hspace=0.1)

        j  = 0  
        counter = 1
        while j < n:
            # for drawing top row
            ax_img = fig.add_subplot(grid[j])
            ax_img.set_xticks([])
            ax_img.set_yticks([])

            ax_img_attn = fig_attn.add_subplot(grid[j])
            ax_img_attn.set_xticks([])
            ax_img_attn.set_yticks([])

            if j == 0:
                ax_img.imshow(default_loader(qimages[i]))
                bbx = bbxs[i]
                qim_cropped = default_loader(qimages[i]).crop(bbx)
                ax_img_attn.imshow(qim_cropped)
                attn = get_attn(model_retr, qim_cropped)
                ax_img_attn.imshow(attn, cmap='jet', alpha=.65)

                left, bot = bbx[0], bbx[1]
                b_h, b_w = bbx[3] - bbx[1], bbx[2] - bbx[0]

                for spine in ax_img.spines.values():
                    spine.set_edgecolor('black')
                    spine.set_linewidth(1)

                rect = patches.Rectangle((left, bot), b_w, b_h, fill=False, linewidth=1.5, edgecolor='y', facecolor='None')
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

                db_img = default_loader(images[db_id])
                ax_img.imshow(db_img)
                ax_img_attn.imshow(db_img)

                attn = get_attn(model_retr, db_img)
                ax_img_attn.imshow(attn, cmap='jet', alpha=.65)

                for spine in ax_img.spines.values():
                    spine.set_edgecolor(edgecolor)
                    spine.set_linewidth(2)
                counter +=1 
                j += 1

        summary.add_figure('/' + protocol + '/ranks/', fig, global_step=i+1)
        summary.add_figure('/' + protocol + '/attns/', fig_attn, global_step=i+1)