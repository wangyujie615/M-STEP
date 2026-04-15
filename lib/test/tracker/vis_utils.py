import numpy as np


############## used for visulize eliminated tokens #################
def get_keep_indices(decisions):
    keep_indices = []
    for i in range(3):
        if i == 0:
            keep_indices.append(decisions[i])
        else:
            keep_indices.append(keep_indices[-1][decisions[i]])
    return keep_indices


def gen_masked_tokens(tokens, indices, alpha=0.2):
    # indices = [i for i in range(196) if i not in indices]
    indices = indices[0].astype(int)
    tokens = tokens.copy()
    tokens[indices] = alpha * tokens[indices] + (1 - alpha) * 255
    return tokens


def recover_image(tokens, H, W, Hp, Wp, patch_size):
    # image: (C, 196, 16, 16)
    image = tokens.reshape(Hp, Wp, patch_size, patch_size, 3).swapaxes(1, 2).reshape(H, W, 3)
    return image


def pad_img(img):
    height, width, channels = img.shape
    im_bg = np.ones((height, width + 8, channels)) * 255
    im_bg[0:height, 0:width, :] = img
    return im_bg


def gen_visualization(image, mask_indices, patch_size=16):
    # image [224, 224, 3]
    # mask_indices, list of masked token indices

    # mask mask_indices need to cat
    # mask_indices = mask_indices[::-1]
    num_stages = len(mask_indices)
    for i in range(1, num_stages):
        mask_indices[i] = np.concatenate([mask_indices[i-1], mask_indices[i]], axis=1)

    # keep_indices = get_keep_indices(decisions)
    image = np.asarray(image)
    H, W, C = image.shape
    Hp, Wp = H // patch_size, W // patch_size
    image_tokens = image.reshape(Hp, patch_size, Wp, patch_size, 3).swapaxes(1, 2).reshape(Hp * Wp, patch_size, patch_size, 3)

    stages = [
        recover_image(gen_masked_tokens(image_tokens, mask_indices[i]), H, W, Hp, Wp, patch_size)
        for i in range(num_stages)
    ]
    imgs = [image] + stages
    imgs = [pad_img(img) for img in imgs]
    viz = np.concatenate(imgs, axis=1)
    return viz

def vis_attn_maps(attn_weights, q_w, k_w, skip_len, x1, x2, x1_title, x2_title, save_path='.', idxs=None):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    shape1 = [q_w, q_w]
    shape2 = [k_w, k_w]

    attn_weights_mean = []
    for attn in attn_weights:
        attn_weights_mean.append(attn[..., skip_len:(skip_len+k_w**2)].mean(dim=1).squeeze().reshape(shape1+shape2).cpu())

    # downsampling factor
    fact = 32

    # let's select 4 reference points for visualization
    # idxs = [(32, 32), (64, 64), (32, 96), (96, 96), ]
    if idxs is None:
        idxs = [(64, 64)]

    block_num=0
    idx_o = idxs[0]
    for attn_weight in attn_weights_mean:
        fig = plt.figure(constrained_layout=False, figsize=(5, 5), dpi=160)
        fig.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0)
        ax = fig.add_subplot(111)
        idx = (idx_o[0] // fact, idx_o[1] // fact)
        ax.imshow(attn_weight[..., idx[0], idx[1]], cmap='cividis', interpolation='nearest')
        ax.axis('off')
        # ax.set_title(f'Stage2-Block{block_num}')
        plt.savefig(save_path + '/Stage2-Block{}_attn_weight.png'.format(block_num))
        plt.close()
        block_num += 1

    fig = plt.figure(constrained_layout=False, figsize=(5, 5), dpi=160)
    fig.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0)
    x2_ax = fig.add_subplot(111)
    x2_ax.imshow(x2)
    x2_ax.axis('off')
    plt.savefig(save_path + '/{}.png'.format(x2_title))
    plt.close()

    # the reference points as red circles
    fig = plt.figure(constrained_layout=False, figsize=(5, 5), dpi=160)
    fig.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0)
    x1_ax = fig.add_subplot(111)
    x1_ax.imshow(x1)
    for (y, x) in idxs:
        # scale = im.height / img.shape[-2]
        x = ((x // fact) + 0.5) * fact
        y = ((y // fact) + 0.5) * fact
        x1_ax.add_patch(plt.Circle((x, y), fact // 2, color='r'))
        # x1_ax.set_title(x1_title)
        x1_ax.axis('off')
    plt.savefig(save_path+'/{}.png'.format(x1_title))
    plt.close()

    del attn_weights_mean
