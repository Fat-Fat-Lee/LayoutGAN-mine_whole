import pickle
import argparse
from pathlib import Path

import torch
from torch_geometric.data import DataLoader
from torch_geometric.utils import to_dense_batch

from util import set_seed, convert_layout_to_image
from data import get_dataset
from model.layoutganpp import Generator

#python generate.py F:\GradeThree\YuQianLab\GraduateDesign\const_layout_remote\pretrained\model_best.pth.tar --out_path output/generated_layouts_ricoTest.pkl --num_save 5
#python generate.py F:\GradeThree\YuQianLab\GraduateDesign\const_layout_remote\pretrained\layoutganpp_rico.pth.tar --out_path output/generated_layouts_ricoTest.pkl --num_save 5

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt_path', type=str, help='checkpoint path')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('-o', '--out_path', type=str,
                        default='output/generated_layouts.pkl',
                        help='output pickle path')
    parser.add_argument('--num_save', type=int, default=0,
                        help='number of layouts to save as images')
    parser.add_argument('--seed', type=int, help='manual seed')
    parser.add_argument('--ifnGPUs', type=int, default=1)#by ljw 20230213 多gpu训练
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

    # load test dataset
    dataset = get_dataset(train_args['dataset'], 'test')
    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            num_workers=0,
                            pin_memory=True,
                            shuffle=False)
    if train_args['dataset'] == 'rico' or train_args['dataset'] == 'ricoTest':
        num_label = 13
    else:
        num_label = dataset.num_classes
    # -------------by ljw 20221109---------------#
    # to do
    if train_args['dataset'] == 'ricoTest':
        num_ricoLabel = dataset.num_ricoClasses
    else:
        num_ricoLabel = -1
        # setup model and load state
    netG = Generator(train_args['latent_size'], num_label,
                     d_model=train_args['G_d_model'],
                     nhead=train_args['G_nhead'],
                     num_layers=train_args['G_num_layers'],
                     num_ricoLabel=num_ricoLabel
                    ).eval().to(device)
    if args.ifnGPUs==1:
        netG = torch.nn.DataParallel(netG).cuda()#by ljw 20230213 多gpu训练
    # -------------------------------------------#

    netG.load_state_dict(ckpt['netG'])

    results = []
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)

            label, mask = to_dense_batch(data.y, data.batch)
            padding_mask = ~mask
            z = torch.randn(label.size(0), label.size(1),
                            train_args['latent_size'], device=device)
            # --------by ljw 20221109------------------#

            # to do加入ricoLabel映射
            if train_args['dataset'] == 'ricoTest':
                ricoLabel, __ = to_dense_batch(data.ricoLabel, data.batch)
                bbox = netG(z, label, padding_mask, ricoLabel)
            else:
                ricoLabel = torch.tensor(-1)
                bbox = netG(z, label, padding_mask)
            # -----------------------------------------#


            for j in range(bbox.size(0)):
                mask_j = mask[j]
                attr = data.attr[j]
                b = bbox[j][mask_j].cpu().numpy()
                l = label[j][mask_j].cpu().numpy()


                if len(results) < args.num_save:
                    convert_layout_to_image(
                        b, l, dataset.colors, (120, 80)
                    ).save(out_dir / f'generated_{len(results)}.png')

                # if train_args['dataset'] == 'ricoTest':
                #     r_l=ricoLabel[j][mask_j].cpu().numpy()#by ljw 20230103
                #     results.append((b,l,r_l))  # by ljw 20230103
                # else:
                results.append((b, l,attr)) # by ljw 20230103


    # save results
    with out_path.open('wb') as fb:
        pickle.dump(results, fb)
    print('Generated layouts are saved at:', args.out_path)


if __name__ == '__main__':
    main()
