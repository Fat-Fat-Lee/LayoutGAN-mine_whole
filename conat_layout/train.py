import os
import argparse
os.environ['OMP_NUM_THREADS'] = '1'  # noqa

import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.utils import to_dense_batch
from torch.utils.tensorboard import SummaryWriter

from data import get_dataset
from metric import LayoutFID, compute_maximum_iou
from model.layoutganpp import Generator, Discriminator
from data.util import LexicographicSort, HorizontalFlip
from util import init_experiment, save_image, save_checkpoint
#python train.py --dataset rico --batch_size 2 --iteration 200000 --latent_size 8 --lr 5e-06 --G_d_model 256 --G_nhead 4 --G_num_layers 8 --D_d_model 256 --D_nhead 4 --D_num_layers 8
# python train.py --dataset ricoTest --batch_size 64 --iteration 200000 --latent_size 8 --lr 5e-06 --G_d_model 256 --G_nhead 4 --G_num_layers 8 --D_d_model 256 --D_nhead 4 --D_num_layers 8


def main():
    CUDA_LAUNCH_BLOCKING = "1"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--name', type=str, default='',
                        help='experiment name')
    parser.add_argument('--dataset', type=str, default='ricoTest',
                        choices=['rico', 'publaynet', 'magazine','ricoTest'],#by ljw 20221103
                        help='dataset name')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--iteration', type=int, default=int(2e+5),
                        help='number of iterations to train for')
    parser.add_argument('--seed', type=int, help='manual seed')

    # General
    parser.add_argument('--latent_size', type=int, default=4,
                        help='latent size')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='learning rate')
    parser.add_argument('--aug_flip', action='store_true',
                        help='use horizontal flip for data augmentation.')

    # Generator
    parser.add_argument('--G_d_model', type=int, default=256,
                        help='d_model for generator')
    parser.add_argument('--G_nhead', type=int, default=4,
                        help='nhead for generator')
    parser.add_argument('--G_num_layers', type=int, default=8,
                        help='num_layers for generator')

    # Discriminator
    parser.add_argument('--D_d_model', type=int, default=256,
                        help='d_model for discriminator')
    parser.add_argument('--D_nhead', type=int, default=4,
                        help='nhead for discriminator')
    parser.add_argument('--D_num_layers', type=int, default=8,
                        help='num_layers for discriminator')

    args = parser.parse_args()
    print(args)

    out_dir = init_experiment(args, "LayoutGAN++")
    writer = SummaryWriter(out_dir)


    # load dataset
    transforms = [LexicographicSort()]
    if args.aug_flip:
        transforms = [T.RandomApply([HorizontalFlip()], 0.5)] + transforms

    train_dataset = get_dataset(args.dataset, 'train',
                                transform=T.Compose(transforms))

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  num_workers=0,
                                  pin_memory=True,
                                  shuffle=True)

    val_dataset = get_dataset(args.dataset, 'val')
    val_dataloader = DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                num_workers=0,
                                pin_memory=True,
                                shuffle=False)

    if args.dataset == 'ricoTest' or args.dataset == 'ricoTest':
        num_label = 13
    else:
        num_label = train_dataset.num_classes
    #-------------by ljw 20221102---------------#
    #to do
    if args.dataset=='ricoTest':
        num_ricoLabel=train_dataset.num_ricoClasses
    else:
        num_ricoLabel=-1
    #-------------------------------------------#
    # load pre-trained LayoutNet
    tmpl = './output/ricoTest/LayoutGAN++/20230213110437510104/model_best.pth.tar'
    #
    # # --------------by ljw 20221103------------------#
    # # 加载部分预训练参数，其余新加的修改部分不加载
    ckpt = torch.load(tmpl.format(args.dataset), map_location=device)
    train_args = ckpt['args']
    # setup model
    # netG = Generator(train_args['latent_size'], num_label,
    #                  d_model=train_args['G_d_model'],
    #                  nhead=train_args['G_nhead'],
    #                  num_layers=train_args['G_num_layers'],
    #                  num_ricoLabel=num_ricoLabel#by ljw 20221102,加入num_ricoLabel
    #                  ).to(device)
    #
    # netG.load_state_dict(ckpt['netG'],strict=False)#by ljw 20230205
    # # print(num_label)
    #
    # netD = Discriminator(num_label,
    #                      d_model=train_args['D_d_model'],
    #                      nhead=train_args['D_nhead'],
    #                      num_layers=train_args['D_num_layers'],
    #                      num_ricoLabel=num_ricoLabel  # by ljw 20221212,加入num_ricoLabel
    #                      ).to(device)
    # netD.load_state_dict(ckpt['netD'],strict=False)  # by ljw 20230205

    netG = Generator(args.latent_size, num_label,
                     d_model=args.G_d_model,
                     nhead=args.G_nhead,
                     num_layers=args.G_num_layers,
                     num_ricoLabel=num_ricoLabel  # by ljw 20221212,加入num_ricoLabel
                     ).to(device)

    netD = Discriminator(num_label,
                         d_model=args.D_d_model,
                         nhead=args.D_nhead,
                         num_layers=args.D_num_layers,
                         num_ricoLabel=num_ricoLabel  # by ljw 20221212,加入num_ricoLabel
                         ).to(device)




    # prepare for evaluation
    fid_train = LayoutFID("ricoTest", device)
    fid_val = LayoutFID("ricoTest", device)

    fixed_label = None
    val_layouts = [(data.x.numpy(), data.y.numpy()) for data in val_dataset]

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr)
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr)

    iteration = 0
    last_eval, best_iou = -1e+8, -1e+8
    max_epoch = args.iteration * args.batch_size / len(train_dataset)
    max_epoch = int(torch.ceil(torch.tensor(max_epoch)).item())

    train_index=0
    for epoch in range(max_epoch):
        #--------by ljw 20230205--------#
        #控制训练速率
        # if train_index==30:
        #     train_index=0
        # if train_index==0:
        #     netD.train()
        # train_index += 1

        netG.train(),netD.train()
        #------------------------------#
        for i, data in enumerate(train_dataloader):
            data = data.to(device)

            label, mask = to_dense_batch(data.y, data.batch)
            bbox_real, _ = to_dense_batch(data.x, data.batch)


            padding_mask = ~mask
            z = torch.randn(label.size(0), label.size(1),
                            args.latent_size, device=device)

            # Update G network
            netG.zero_grad()
            # --------by ljw 20221102------------------#

            # to do加入ricoLabel映射
            if args.dataset == 'ricoTest':
                ricoLabel,__=to_dense_batch(data.ricoLabel, data.batch)
                bbox_fake = netG(z, label, padding_mask,ricoLabel)
            else:
                ricoLabel = torch.tensor(-1)
                bbox_fake = netG(z, label, padding_mask)
            # -----------------------------------------#

            #-----------by ljw 20221212-----------------#
            D_fake = netD(bbox_fake, label, padding_mask,ricoLabel=ricoLabel)
            #-------------------------------------------#
            loss_G = F.softplus(-D_fake).mean()
            loss_G.backward()
            optimizerG.step()

            # Update D network
            netD.zero_grad()

            # -----------by ljw 20221212-----------------#
            D_fake = netD(bbox_fake.detach(), label, padding_mask,ricoLabel=ricoLabel)

            # -------------------------------------------#
            loss_D_fake = F.softplus(D_fake).mean()

            D_real, logit_cls, bbox_recon = netD(bbox_real, label, padding_mask, reconst=True,ricoLabel=ricoLabel)#by ljw 20221212
            loss_D_real = F.softplus(-D_real).mean()
            loss_D_recl = F.cross_entropy(logit_cls, data.y)
            loss_D_recb = F.mse_loss(bbox_recon, data.x)

            loss_D = loss_D_real + loss_D_fake
            loss_D += loss_D_recl + 10 * loss_D_recb
            loss_D.backward()
            optimizerD.step()

            # ------------------------by ljw 20221103------------------#
            # to do加入ricoLabel映射
            # if args.dataset == 'ricoTest':
            #     fid_train.collect_features(bbox=bbox_fake,label= label, padding_mask=padding_mask,
            #                                ricoLabel=ricoLabel)
            #     fid_train.collect_features(bbox=bbox_real,label= label, padding_mask=padding_mask,
            #                                ricoLabel=ricoLabel,real=True)
            # else:

            fid_train.collect_features(bbox=bbox_fake,label= label, padding_mask=padding_mask)
            fid_train.collect_features(bbox=bbox_real,label= label, padding_mask=padding_mask,
                                           real=True)
            # ------------------------------------------------------------#


            if iteration % 50 == 0:
                D_real = torch.sigmoid(D_real).mean().item()
                D_fake = torch.sigmoid(D_fake).mean().item()
                loss_D, loss_G = loss_D.item(), loss_G.item()
                loss_D_fake, loss_D_real = loss_D_fake.item(), loss_D_real.item()
                loss_D_recl, loss_D_recb = loss_D_recl.item(), loss_D_recb.item()

                print('\t'.join([
                    f'[{epoch}/{max_epoch}][{i}/{len(train_dataloader)}]',
                    f'Loss_D: {loss_D:E}', f'Loss_G: {loss_G:E}',
                    f'Real: {D_real:.3f}', f'Fake: {D_fake:.3f}',
                ]))

                # add data to tensorboard
                tag_scalar_dict = {'real': D_real, 'fake': D_fake}
                writer.add_scalars('Train/D_value', tag_scalar_dict, iteration)
                writer.add_scalar('Train/Loss_D', loss_D, iteration)
                writer.add_scalar('Train/Loss_D_fake', loss_D_fake, iteration)
                writer.add_scalar('Train/Loss_D_real', loss_D_real, iteration)
                writer.add_scalar('Train/Loss_D_recl', loss_D_recl, iteration)
                writer.add_scalar('Train/Loss_D_recb', loss_D_recb, iteration)
                writer.add_scalar('Train/Loss_G', loss_G, iteration)

            if iteration % 5000 == 0:
                out_path = out_dir / f'real_samples.png'
                if not out_path.exists():
                    save_image(bbox_real, label, mask,
                               train_dataset.colors, out_path)

                if fixed_label is None:
                    fixed_label = label
                    fixed_ricoLabel = ricoLabel#by ljw 20221104
                    fixed_z = z
                    fixed_mask = mask

                with torch.no_grad():
                    netG.eval()
                    out_path = out_dir / f'fake_samples_{iteration:07d}.png'
                    bbox_fake = netG(fixed_z, fixed_label, ~fixed_mask,ricoLabel)#加上ricoLabel
                    save_image(bbox_fake, fixed_label, fixed_mask,
                               train_dataset.colors, out_path)
                    netG.train()

            iteration += 1
            # torch.cuda.empty_cache()  # by ljw 20220919
            # torch.cuda.empty_cache()  # by ljw 20220919
            # torch.cuda.empty_cache()  # by ljw 20220919
            # torch.cuda.empty_cache()  # by ljw 20220919

        fid_score_train = fid_train.compute_score()

        if epoch != max_epoch - 1:
            if iteration - last_eval < 1e+4:
                continue

        # validation
        last_eval = iteration
        fake_layouts = []
        netG.eval(), netD.eval()
        with torch.no_grad():
            for i, data in enumerate(val_dataloader):
                data = data.to(device)
                label, mask = to_dense_batch(data.y, data.batch)
                bbox_real, _ = to_dense_batch(data.x, data.batch)
                padding_mask = ~mask
                z = torch.randn(label.size(0), label.size(1),
                                args.latent_size, device=device)

                # --------by ljw 20221108------------------#

                # to do加入ricoLabel映射
                if args.dataset == 'ricoTest':
                    ricoLabel, __ = to_dense_batch(data.ricoLabel, data.batch)
                    bbox_fake = netG(z, label, padding_mask, ricoLabel)
                else:
                    ricoLabel = torch.tensor(-1)
                    bbox_fake = netG(z, label, padding_mask)
                # -----------------------------------------#

                # ------------------------by ljw 20221108------------------#
                # to do加入ricoLabel映射
                # if args.dataset == 'ricoTest':
                #         fid_val.collect_features(bbox=bbox_fake, label=label, padding_mask=padding_mask,
                #                                    ricoLabel=ricoLabel)
                #         fid_val.collect_features(bbox=bbox_real, label=label, padding_mask=padding_mask,
                #                                    ricoLabel=ricoLabel, real=True)
                # else:

                fid_val.collect_features(bbox=bbox_fake, label=label, padding_mask=padding_mask)
                fid_val.collect_features(bbox=bbox_real, label=label, padding_mask=padding_mask,
                                                   real=True)
                # ------------------------------------------------------------#



                # collect generated layouts
                for j in range(label.size(0)):
                    _mask = mask[j]
                    b = bbox_fake[j][_mask].cpu().numpy()
                    l = label[j][_mask].cpu().numpy()
                    fake_layouts.append((b, l))

        fid_score_val = fid_val.compute_score()
        max_iou_val = compute_maximum_iou(val_layouts, fake_layouts)

        writer.add_scalar('Epoch', epoch, iteration)
        tag_scalar_dict = {'train': fid_score_train, 'val': fid_score_val}
        writer.add_scalars('Score/Layout FID', tag_scalar_dict, iteration)
        writer.add_scalar('Score/Maximum IoU', max_iou_val, iteration)

        # do checkpointing
        is_best = best_iou < max_iou_val
        best_iou = max(max_iou_val, best_iou)

        save_checkpoint({
            'args': vars(args),
            'epoch': epoch + 1,
            'netG': netG.state_dict(),
            'netD': netD.state_dict(),
            'best_iou': best_iou,
            'optimizerG': optimizerG.state_dict(),
            'optimizerD': optimizerD.state_dict(),
        }, is_best, out_dir)

        # torch.cuda.empty_cache()  # by ljw 20220919
        # torch.cuda.empty_cache()  # by ljw 20220919
        # torch.cuda.empty_cache()  # by ljw 20220919
        # torch.cuda.empty_cache()  # by ljw 20220919
        print("epoch/max_epoch",epoch,"/",max_epoch,"  ","is_best",is_best,"best_iou",best_iou,"max_iou_val",max_iou_val)


if __name__ == "__main__":
    main()
