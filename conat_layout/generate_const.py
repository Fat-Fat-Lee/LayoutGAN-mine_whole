import os
os.environ['OMP_NUM_THREADS'] = '1'  # noqa

import pickle
import argparse
import tempfile
import subprocess
from tqdm import tqdm
from pathlib import Path

import torch
import torchvision.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.utils import to_dense_batch

from data import get_dataset
from util import set_seed, convert_layout_to_image
from data.util import AddCanvasElement, AddRelation
from model.layoutganpp import Generator, Discriminator

import clg.const
from clg.auglag import AugLagMethod
from clg.optim import AdamOptimizer, CMAESOptimizer
from metric import compute_violation, compute_Irregular
from torch.utils.data.distributed import DistributedSampler


#pretrained/model_best.pth.tar --const_type relation --out_path output/relation/generated_ricoTestlayouts.pkl --num_save 5
#pretrained/layoutganpp_rico.pth.tar --const_type relation --out_path output/relation/generated_ricolayouts.pkl --num_save 5
#./output/ricoTest/LayoutGAN++/20230223004050561103/model_best.pth.tar --const_type relation --out_path output/relation/generated_ricoTest_relation7772.pkl --num_save 50
def save_gif(out_path, j, netG,
             z_hist, label, mask, padding_mask,
             dataset_colors, canvas_size,ricoLabel,ifRicoTest=False):
    mask = mask[j]
    _j = slice(j, j + 1)

    z_before, z_filtered = None, []
    for z in z_hist:
        if z_before is not None:
            if z_before.eq(z[_j]).all():
                continue
        z_filtered.append(z)
        z_before = z[_j]
    z_filtered += [z] * 2

    with tempfile.TemporaryDirectory() as tempdir:
        for i, z in enumerate(z_filtered):
            if ifRicoTest is True:
                bbox = netG(z[_j], label[_j], padding_mask[_j],ricoLabel=ricoLabel[_j])  #by ljw 20230105
            else:
                bbox = netG(z[_j], label[_j], padding_mask[_j],ricoLabel=-1)#by ljw 20230105
            b = bbox[0][mask].cpu().detach().numpy()#by ljw 20230105
            l = label[0][mask].cpu().detach().numpy()#by ljw 20230105

            convert_layout_to_image(
                b, l, dataset_colors, canvas_size
            ).save(tempdir + f'/{j}_{i:08d}.png')

        subprocess.run(['convert', '-delay', '50',
                        tempdir + f'/{j}_*.png', str(out_path)])


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('ckpt_path', type=str, help='checkpoint path')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('-o', '--out_path', type=str,
                        default='output/generated_layouts.pkl',
                        help='output pickle path')
    parser.add_argument('--num_save', type=int, default=0,
                        help='number of layouts to save as images')
    parser.add_argument('--seed', type=int, help='manual seed')

    # CLG specific options
    parser.add_argument('--const_type', type=str,
                        default='beautify', help='constraint type',
                        choices=['beautify', 'relation'])
    parser.add_argument('--optimizer', type=str,
                        default='CMAES', help='inner optimizer',
                        choices=['Adam', 'CMAES'])
    parser.add_argument('--rel_ratio', type=float, default=0.1,
                        help='ratio of relational constraints')
    parser.add_argument("--local_rank", type=int)
    parser.add_argument('--ifnGPUs', type=int, default=1)  # by ljw 20230213 多gpu训练

    args = parser.parse_args()

    if args.seed is not None:
        set_seed(args.seed)

    out_path = Path(args.out_path)
    out_dir = out_path.parent
    out_dir.mkdir(exist_ok=True, parents=True)

    # load checkpoint
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt_path, map_location=device)
    train_args = ckpt['args']

    # setup transforms and constraints
    transforms = [AddCanvasElement()]
    if args.const_type == 'relation':
        transforms += [AddRelation(args.seed, args.rel_ratio)]
        constraints = clg.const.relation
    else:
        constraints = clg.const.beautify

    # load test dataset
    dataset = get_dataset(train_args['dataset'], 'randn',
                          T.Compose(transforms))#by ljw 20230222 加载随机数据集
    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            num_workers=0,
                            pin_memory=True,
                            shuffle=False)
    num_label = dataset.num_classes

    # -------------by ljw 20221109---------------#
    # to do
    if train_args['dataset'] == 'ricoTest':
        num_ricoLabel = dataset.num_ricoClasses
        netG = Generator(train_args['latent_size'], num_label,
                         d_model=train_args['G_d_model'],
                         nhead=train_args['G_nhead'],
                         num_layers=train_args['G_num_layers'],
                         num_ricoLabel=num_ricoLabel
                         ).eval().to(device)
        netD = Discriminator(num_label,
                             d_model=train_args['D_d_model'],
                             nhead=train_args['D_nhead'],
                             num_layers=train_args['D_num_layers'],
                             num_ricoLabel=num_ricoLabel
                             ).eval().requires_grad_(False).to(device)
    else:
        num_ricoLabel = -1
        netG = Generator(train_args['latent_size'], num_label,
                         d_model=train_args['G_d_model'],
                         nhead=train_args['G_nhead'],
                         num_layers=train_args['G_num_layers'],
                         ).eval().to(device)
        netD = Discriminator(num_label,
                             d_model=train_args['D_d_model'],
                             nhead=train_args['D_nhead'],
                             num_layers=train_args['D_num_layers'],
                             ).eval().requires_grad_(False).to(device)
        # setup model and load state

    if args.ifnGPUs==1:
        netG = torch.nn.DataParallel(netG).cuda()#by ljw 20230213 多gpu训练
        netD = torch.nn.DataParallel(netD).cuda()  # by ljw 20230213 多gpu训练
    # -------------------------------------------#
    netG.load_state_dict(ckpt['netG'])
    netD.load_state_dict(ckpt['netD'])

    # setup optimizers
    if args.optimizer == 'CMAES':
        inner_optimizer = CMAESOptimizer(seed=args.seed)
    else:
        inner_optimizer = AdamOptimizer()

    # -------------by ljw 20221110---------------#
    if train_args['dataset'] == 'ricoTest':
        optimizer = AugLagMethod(netG, netD, inner_optimizer, constraints,ifRicoTest=True)
    else:
        optimizer = AugLagMethod(netG, netD, inner_optimizer, constraints,ifRicoTest=False)
    #-------------------------------------------#

    results, violation = [], []
    for data in tqdm(dataloader, ncols=100):
        data = data.to(device)
        label_c, mask_c = to_dense_batch(data.y, data.batch)
        label = torch.relu(label_c[:, 1:] - 1)

        bbox_real, _ = to_dense_batch(data.x, data.batch)  # by ljw 20230222

        mask = mask_c[:, 1:]
        padding_mask = ~mask

        z = torch.randn(label.size(0), label.size(1),
                        train_args['latent_size'],
                        device=device)
        # --------by ljw 20221109------------------#

        # to do加入ricoLabel映射
        if train_args['dataset'] == 'ricoTest':
            ricoLabel_c, __ = to_dense_batch(data.ricoLabel, data.batch)
            ricoLabel = torch.relu(ricoLabel_c[:, 1:] - 1)
            bbox = netG(z, label, padding_mask, ricoLabel)
        else:
            bbox = netG(z, label, padding_mask)
        # -----------------------------------------#
        z_hist = [z]
        for z in optimizer.generator(z, data):
            if len(results) < args.num_save:
                z_hist.append(z)


        if args.const_type == 'relation':
            canvas = optimizer.bbox_canvas.to(bbox)
            canvas = canvas.expand(bbox.size(0), -1, -1)
            bbox_flatten = torch.cat([canvas, bbox], dim=1)[mask_c]
            v = compute_violation(bbox_flatten, data)
            violation += v[~v.isnan()].tolist()

        if len(results) < args.num_save:
            bbox_init = bbox_real[:, 1:]


        for j in range(bbox.size(0)):
            mask_j = mask[j]

            attr = data.attr[j]
            b = bbox[j][mask_j].cpu().detach().numpy()#by ljw 20230104
            l = label[j][mask_j].cpu().detach().numpy()#by ljw 20230104

            if len(results) < args.num_save:
                b_init = bbox_init[j][mask_j].cpu().detach().numpy()  # by ljw 20230104
                l_init = label[j][mask_j].cpu().detach().numpy()  # by ljw 20230104
                out_path = out_dir / f'initial_{len(results)}.png'
                convert_layout_to_image(
                    bbox_init[j][mask_j].cpu().detach().numpy(),
                    l, dataset.colors, (2560, 1440)
                ).save(out_path)
                tmp_point, minus_reason = compute_Irregular(b_init, l_init)  # by ljw 20230221
                print(len(results))
                print(minus_reason)  # by ljw 20230221

                out_path = out_dir / f'optimized_{len(results)}.png'
                convert_layout_to_image(
                    b, l, dataset.colors, (2560, 1440)
                ).save(out_path)

                out_path = out_dir / f'optimizing_{len(results)}.gif'


                results.append((b, l,attr))
                #---------------------------------------------------------#



    if args.const_type == 'relation':
        violation = sum(violation) / len(violation)
        print(f'Relation violation: {violation:.2%}')

    # save results
    with Path(args.out_path).open('wb') as fb:
        pickle.dump(results, fb)
    print('Generated layouts are saved at:', args.out_path)


if __name__ == '__main__':
    main()
