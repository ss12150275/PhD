# -*- coding:utf-8 -*-
from dataset.dataset_lits import Lits_DataSet
from torch.utils.data import *
import torch
import torch.optim as optim

import config

from models.kiunet2 import *
from utils import logger, init_util, metrics,common


def val(model, val_loader, epoch, logger):
    model.eval()
    val_loss = 0
    val_dice0 = 0
    val_dice1 = 0
    val_dice2 = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.float(), target.float()
            data, target = data.to(device), target.to(device)
            output = model(data)

            loss = metrics.DiceMeanLoss()(output, target)
            # loss = nn.CrossEntropyLoss()(output, target)
            dice0 = metrics.dice(output, target, 0)
            dice1 = metrics.dice(output, target, 1)
            # dice2 = metrics.dice(output, target, 2)

            val_loss += float(loss)
            val_dice0 += float(dice0)
            val_dice1 += float(dice1)
            # val_dice2 += float(dice2)

    val_loss /= len(val_loader)
    val_dice0 /= len(val_loader)
    val_dice1 /= len(val_loader)
    # val_dice2 /= len(val_loader)

    logger.scalar_summary('val_loss', val_loss, epoch)
    logger.scalar_summary('val_dice0', val_dice0, epoch)
    logger.scalar_summary('val_dice1', val_dice1, epoch)
    logger.scalar_summary('val_dice2', val_dice2, epoch)
    print('\nVal set: Average loss: {:.6f}, dice0: {:.6f}\tdice1: {:.6f}\tdice2: {:.6f}\t\n'.format(
        val_loss, val_dice0, val_dice1, val_dice2))


def train(model, train_loader, optimizer, epoch, logger):
    model.train()
    train_loss = 0
    train_dice0 = 0
    train_dice1 = 0
    train_dice2 = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        print('batch_idx',batch_idx)
        data, target = data.float(), target.float()
        data, target = data.to(device), target.to(device)
        print('the shape of output',data.shape)
        output = model(data)
        # print('the shape of output',data.shape)
        # print('the shape of target', target.shape)

        optimizer.zero_grad()

        # loss = nn.CrossEntropyLoss()(output,target)
        # loss=metrics.SoftDiceLoss()(output,target)
        # loss=nn.MSELoss()(output,target)
        loss = metrics.DiceMeanLoss()(output, target)
        # loss=metrics.WeightDiceLoss()(output,target)
        # loss=metrics.cross_entropy_3D(output,target)
        # loss = metrics.Dice_coff_loss()(output, target)
        loss.backward()
        optimizer.step()

        train_loss = loss
        train_dice = metrics.dice_coeff(output, target)
        train_dice0 = metrics.dice(output, target, 0)
        train_dice1 = metrics.dice(output, target, 1)
        print(
            'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tdice0: {:.6f}\tdice1: {:.6f}\tdice2: {:.6f}\tT: {:.6f}\tP: {:.6f}\tTP: {:.6f}'.format(
                epoch, batch_idx+1, len(train_loader),
                100. * (batch_idx+1) / len(train_loader), loss.item(),
                train_dice0, train_dice1, train_dice2,
                metrics.T(output, target), metrics.P(output, target), metrics.TP(output, target)))
        print('train_dice',train_dice)

    logger.scalar_summary('train_loss', float(train_loss), epoch)
    logger.scalar_summary('train_dice0', float(train_dice0), epoch)
    logger.scalar_summary('train_dice1', float(train_dice1), epoch)
    logger.scalar_summary('train_dice2', float(train_dice2), epoch)


if __name__ == '__main__':
    args = config.args
    device = torch.device('cpu' if args.cpu else 'cuda')
    # data info
    print('读取训练集')
    train_set = Lits_DataSet(args.crop_size, args.resize_scale, args.dataset_path, mode='train')
    val_set = Lits_DataSet(args.crop_size, args.resize_scale, args.valset_path, mode='val')
    print('读取完测试集')
    train_loader = DataLoader(dataset=train_set,batch_size=args.batch_size,num_workers=1, shuffle=True)
    val_loader = DataLoader(dataset=val_set,batch_size=args.batch_size,num_workers=1, shuffle=True)
    # model info
    # model = UNet(1, [32, 48, 64, 96, 128], 2, net_mode='3d',conv_block=RecombinationBlock).to(device)
    # model = AttU_Net(1, [32, 48, 64, 96, 128], 2).to(device)
    # model = Atten_UNet(1, [16, 32, 64, 128,256], 2,net_mode='3d').to(device)
    # model = message_passing(1, [16, 32, 64, 128, 256], 2, net_mode='3d').to(device)
    # model = asppUNetcoor1(1, [16, 32, 64, 128, 256], 2, net_mode='3d').to(device)
    model = kiunet3d().to(device)
    # model = NestedUNet(1,2, net_mode='3d').to(device)
    # model = ASPP_UNet(1,[32, 64, 128, 256], 2, net_mode='3d').to(device)
    # model = resUNet3D(1, [32, 64, 128,256], 2, net_mode='3d').to(device)
    # model = Nonlocal_UNet(1, [4, 8, 16, 32, 64], 2,net_mode='3d').to(device)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999))
    init_util.print_network(model)
    # model = nn.DataParallel(model, device_ids=[0,1])  # multi-GPU

    logger = logger.Logger('./output/{}'.format(args.save))
    for epoch in range(1, args.epochs + 1):
        common.adjust_learning_rate(optimizer, epoch, args)
        print(epoch)
        train(model, train_loader, optimizer, epoch, logger)
        if epoch % 10 == 0:
            val(model, val_loader, epoch, logger)
            torch.save(model, './output/{}/state{}.pkl'.format(args.save,epoch))  # Save model with parameters
        # torch.save(model.state_dict(), './output/{}/param.pkl'.format(args.save))  # Only save parameters