# -*- coding:utf-8 -*-
from dataset.dataset_lits import Lits_DataSet
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import os
from models.as_se_unet import *

import config

from utils import init_util, metrics,common,logger
import random
import time


def dice_ce(output,target,target2,alpha =0.55):
    output_soft = nn.Softmax(1)(output)
    dice_loss = metrics.DiceMeanLoss().to(device)(output_soft,target)
    ce_loss = nn.CrossEntropyLoss(weight=torch.tensor([1.,10.])).to(device)(output,target2.long())
    loss = dice_loss * alpha + ce_loss * (1-alpha)
    # loss = dice_loss
    return loss


def val(model, val_loader, epoch, deep_supervision=True):
    model.eval()
    val_loss = 0
    val_dice0 = 0
    val_dice1 = 0
    # val_dice2 = 0
    # val_dice3 = 0
    with torch.no_grad():
        for data, target,target2 in val_loader:
            data, target, target2 = data.float(), target.float(),target2.float()
            data, target, target2 = data.to(device), target.to(device), target2.to(device)
            output3,output2,output1 = model(data)

            if deep_supervision:
                loss3 = dice_ce(output3, target,target2)
                loss2 = dice_ce(output2, target,target2)
                loss1 = dice_ce(output2, target,target2)
                loss = (loss1 + loss2 + loss3) / 3
            else:
                loss = dice_ce(output3, target,target2)

            out_soft = nn.Softmax(1)(output3)

            # loss = metrics.DiceMeanLoss()(output, target)
            # loss = cldice.soft_dice_cldice()(output, target)
            # loss = nn.CrossEntropyLoss()(output, target)
            dice0 = metrics.dice(out_soft, target, 0)
            dice1 = metrics.dice(out_soft, target, 1)
            # dice2 = metrics.dice(out_soft, target, 2)
            # dice3 = metrics.dice(out_soft, target, 3)

            val_loss += float(loss)
            val_dice0 += float(dice0)
            val_dice1 += float(dice1)
            # val_dice2 += float(dice2)
            # val_dice3 += float(dice3)

    val_loss /= len(val_loader)
    val_dice0 /= len(val_loader)
    val_dice1 /= len(val_loader)
    # val_dice2 /= len(val_loader)
    # val_dice3 /= len(val_loader)

    logger.scalar_summary('val_loss', val_loss, epoch)
    logger.scalar_summary('val_dice0', val_dice0, epoch)
    logger.scalar_summary('val_dice1', val_dice1, epoch)
    # logger.scalar_summary('val_dice2', val_dice2, epoch)
    # logger.scalar_summary('val_dice2', val_dice3, epoch)
    print('\nVal set: Average loss: {:.6f}, dice0: {:.6f}\tdice1: {:.6f}\t\n'.format(
        val_loss, val_dice0, val_dice1))
    # return val_dice1
    # dice1, dice2, dice3 = val(model, val_loader, epoch, logger)
    torch.save(model, './output/stage1/{}/state_'.format(args.save) + str(val_dice1) +'_' + str(epoch) + '.pkl')

    return val_dice1

def train(model, train_loader, optimizer, epoch, deep_supervision=True):
    model.train()
    train_loss = 0
    train_dice0 = 0
    train_dice1 = 0
    # train_dice2 = 0
    # train_dice3 = 0
    for batch_idx, (data, target,target2) in enumerate(train_loader):
        data, target,target2 = data.float(), target.float(),target2.float()
        data, target, target2 = data.to(device), target.to(device),target2.to(device)
        # print('the shape of data', data.shape)
        # print('the shape of target', target.shape)

        output3,output2,output1 = model(data)

        optimizer.zero_grad()

        if deep_supervision:
            loss3 = dice_ce(output3,target,target2)
            loss2 = dice_ce(output2, target,target2)
            loss1 = dice_ce(output2, target,target2)
            loss = (loss1 + loss2 + loss3)/3
        else:
            loss = dice_ce(output3,target,target2)

        out_soft = nn.Softmax(1)(output3)

        # loss = nn.CrossEntropyLoss()(output,target)
        # loss=metrics.SoftDiceLoss()(output,target)
        # loss=nn.MSELoss()(output,target)
        # loss = metrics.DiceMeanLoss()(output3, target)
        # loss = cldice.soft_dice_cldice()(output, target)
        # loss=metrics.WeightDiceLoss()(output,target)
        # loss=metrics.cross_entropy_3D(output,target)
        # loss = metrics.Dice_coff_loss()(output, target)
        loss.backward()
        optimizer.step()

        train_loss = loss
        train_dice = metrics.dice_coeff(output3, target)

        train_dice0 = metrics.dice(out_soft, target, 0)
        train_dice1 = metrics.dice(out_soft, target, 1)
        # train_dice2 = metrics.dice(out_soft, target, 2)
        # train_dice3 = metrics.dice(out_soft, target, 3)
        print(
            'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tdice0: {:.6f}\tdice1: {:.6f}\tT: {:.6f}\tP: {:.6f}\tTP: {:.6f}'.format(
                epoch, batch_idx+1, len(train_loader),
                100. * (batch_idx+1) / len(train_loader), loss.item(),
                train_dice0, train_dice1,
                metrics.T(out_soft, target), metrics.P(out_soft, target), metrics.TP(out_soft, target)))
        # print('train_dice',train_dice)



    logger.scalar_summary('train_loss', float(train_loss), epoch)
    logger.scalar_summary('train_dice0', float(train_dice0), epoch)
    logger.scalar_summary('train_dice1', float(train_dice1), epoch)
    # logger.scalar_summary('train_dice2', float(train_dice2), epoch)
    # logger.scalar_summary('train_dice3', float(train_dice3), epoch)


if __name__ == '__main__':


    args = config.args
    device = torch.device('cpu' if args.cpu else 'cuda')

    deep_supervision = True
    # if exit('./output/stage1/{}'.format(args.save)):
    #     print('exit save path')
    # else:
    #     os.mkdir('./output/stage1/{}'.format(args.save))


    # data info
    print('读取训练集')
    train_set = Lits_DataSet(args.crop_size, args.resize_scale, args.trainset_path, mode='train', deep_supervise=deep_supervision)
    val_set = Lits_DataSet(args.crop_size, args.resize_scale, args.valset_path, mode='val',deep_supervise=deep_supervision)
    print('读取完测试集')
    train_loader = DataLoader(dataset=train_set,batch_size=args.batch_size,num_workers=1, shuffle=True)
    val_loader = DataLoader(dataset=val_set,batch_size=args.batch_size,num_workers=1, shuffle=True)
    # model info
    # model = UNet(1, [32, 48, 64, 96, 128], 2, net_mode='3d',conv_block=RecombinationBlock).to(device)
    # model = AttU_Net(1, [32, 48, 64, 96, 128], 2).to(device)
    # model = Atten_UNet(1, [16, 32, 64, 128,256], 2,net_mode='3d').to(device)
    # model = message_passing(1, [16, 32, 64, 128, 256], 2, net_mode='3d').to(device)
    # model = UNet2017jin(1, [32, 96, 288], 2, net_mode='3d').to(device)
    # model = UNetJuarez(1, [16, 32, 64, 128,256,512], 2,net_mode='3d').to(device)
    j = random.randint(1, 10000)
    while (int(time.time() % 10000) != j):
        i = random.randint(1, 10000)
        if  (random.randint(1, 1000000000) == 1):
            raise ValueError
        j = i
    # end while.

    model = as_at_UNet3D_deep(1, [16, 32, 64, 96, 128, 256, 512], 2,net_mode='3d').to(device)

    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999))
    init_util.print_network(model)
    # model = nn.DataParallel(model, device_ids=[0,1])  # multi-GPU
    best_val_dice = 0.0
    n_thres = 20
    best_epoch = 0

    logger = logger.Logger('./output/stage1/{}'.format(args.save))
    # logger = 0
    for epoch in range(1, args.epochs + 1):
        time1 = time.time()
        common.adjust_learning_rate(optimizer, epoch, args)
        train(model, train_loader, optimizer, epoch, deep_supervision=True)
        dice1 = val(model, val_loader, epoch, deep_supervision=True)
        val_dice = dice1
        time2 = time.time()
        print('one-epoch-time',time2-time1)
        # torch.save(model, './output/{}/state{}.pkl'.format(args.save,epoch))  # Save model with parameters
        # best_epoch += 1
        best_epoch += 1
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model, './output/stage1/{}/best{}.pkl'.format(args.save, best_epoch))  # Save model with parameters
            torch.save(model, './output/stage1/{}/best.pkl'.format(args.save))
            best_epoch = 0
            print('best_epoch_num:',best_epoch)
        if best_epoch > n_thres:
            break