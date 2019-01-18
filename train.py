import argparse
import time

from models import *
from utils.datasets import *
from utils.utils import *

from utils import torch_utils

# Import test.py to get mAP after each epoch
import test

DARKNET_WEIGHTS_FILENAME = 'darknet53.conv.74'
DARKNET_WEIGHTS_URL = 'https://pjreddie.com/media/files/{}'.format(DARKNET_WEIGHTS_FILENAME)


def train(
        net_config_path,
        data_config_path,
        img_size=416,
        resume=False,
        epochs=100,
        batch_size=16,
        accumulated_batches=1,
        weights_path='weights',
        multi_scale=False,
        freeze_backbone=True,
        var=0,
        lr=0.001,
        freeze_for=1
):
    device = torch_utils.select_device()
    print("Using device: \"{}\"".format(device))

    if multi_scale:  # pass maximum multi_scale size
        img_size = 608
    else:
        torch.backends.cudnn.benchmark = True

    os.makedirs(weights_path, exist_ok=True)
    latest_weights_file = os.path.join(weights_path, 'latest.pt')
    best_weights_file = os.path.join(weights_path, 'best.pt')

    # Configure run
    data_config = parse_data_config(data_config_path)
    num_classes = int(data_config['classes'])
    train_path = data_config['train']

    # Initialize model
    model = Darknet(net_config_path, img_size)

    # Get dataloader
    dataloader = load_images_and_labels(train_path, batch_size=batch_size, img_size=img_size,
                                        multi_scale=multi_scale, augment=True)

    # lr0 = lr
    if resume:
        checkpoint = torch.load(latest_weights_file, map_location='cpu')

        model.load_state_dict(checkpoint['model'])
        model.to(device).train()

        # Set optimizer
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

        start_epoch = checkpoint['epoch'] + 1
        if checkpoint['optimizer'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_loss = checkpoint['best_loss']

        del checkpoint  # current, saved

    else:
        start_epoch = 0
        best_loss = float('inf')

        # Initialize model with darknet53 weights (optional)
        def_weight_file = os.path.join(weights_path, DARKNET_WEIGHTS_FILENAME)
        if not os.path.isfile(def_weight_file):
            os.system('wget {} -P {}'.format(
                DARKNET_WEIGHTS_URL,
                weights_path))
        assert os.path.isfile(def_weight_file)
        load_weights(model, def_weight_file)

        model.to(device).train()

        # Set optimizer
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    # Set scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[54, 61], gamma=0.1)

    model_info(model)
    t0 = time.time()
    mean_recall, mean_precision = 0.0, 0.0
    for epoch in range(epochs):
        epoch += start_epoch

        print(('%8s%12s' + '%10s' * 11) % ('Epoch', 'Batch', 'x', 'y', 'w', 'h', 'conf', 'cls', 'total', 'P', 'R',
                                           'nTargets', 'time'))

        scheduler.step()

        if freeze_backbone:
            if epoch == start_epoch:
                for i, (name, p) in enumerate(model.named_parameters()):
                    if int(name.split('.')[1]) < 75:  # if layer < 75
                        p.requires_grad = False
            elif (epoch - start_epoch) == freeze_for:
                for i, (name, p) in enumerate(model.named_parameters()):
                    if int(name.split('.')[1]) < 75:  # if layer < 75
                        p.requires_grad = True

        ui = -1
        rloss = defaultdict(float)  # running loss
        optimizer.zero_grad()
        
        for i, (imgs, targets) in enumerate(dataloader):
            if sum([len(x) for x in targets]) < 1:  # if no targets continue
                continue

            # Compute loss, compute gradient, update parameters
            loss = model(imgs.to(device), targets, batch_report=False, var=var)
            loss.backward()

            # accumulate gradient for x batches before optimizing
            if ((accumulated_batches and ((i + 1) % accumulated_batches == 0))) or (i == len(dataloader) - 1):
                # print('applying optimizer-- acc_batches {}, i {}, dataloader {}'.format(accumulated_batches, i, len(dataloader)))
                optimizer.step()
                optimizer.zero_grad()

            # Running epoch-means of tracked metrics
            ui += 1
            for key, val in model.losses.items():
                rloss[key] = (rloss[key] * ui + val) / (ui + 1)

            s = ('%8s%12s' + '%10.3g' * 11) % (
                '%g/%g' % (epoch, epochs - 1), '%g/%g' % (i, len(dataloader) - 1), rloss['x'],
                rloss['y'], rloss['w'], rloss['h'], rloss['conf'], rloss['cls'],
                rloss['loss'], mean_precision, mean_recall, model.losses['nT'], time.time() - t0)
            t0 = time.time()
            print(s)

        # Update best loss
        loss_per_target = rloss['loss'] / rloss['nT']
        if loss_per_target < best_loss:
            best_loss = loss_per_target

        # Save latest checkpoint
        checkpoint = {'epoch': epoch,
                      'best_loss': best_loss,
                      'model': model.state_dict(),
                      'optimizer': optimizer.state_dict()}
        torch.save(checkpoint, latest_weights_file)

        # Save best checkpoint
        if best_loss == loss_per_target:
            print('Saving best yet!')
            os.system('cp {} {}'.format(
                latest_weights_file,
                best_weights_file,
            ))

        # Save backup weights every 5 epochs
        if (epoch > 0) & (epoch % 5 == 0):
            backup_file_name = 'backup{}.pt'.format(epoch)
            backup_file_path = os.path.join(weights_path, backup_file_name)
            os.system('cp {} {}'.format(
                latest_weights_file,
                backup_file_path,
            ))

        # Calculate mAP
        # mAP, R, P = test.test(
        #     net_config_path,
        #     data_config_path,
        #     latest_weights_file,
        #     batch_size=batch_size,
        #     img_size=img_size,
        # )

        # # Write epoch results
        # with open('results.txt', 'a') as file:
        #     file.write(s + '%11.3g' * 3 % (mAP, P, R) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='size of each image batch')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--accumulated-batches', type=int, default=1, help='number of batches before optimizer step')
    parser.add_argument('--data-config', type=str, default='cfg/flower.data', help='path to data config file')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-flower.cfg', help='cfg file path')
    parser.add_argument('--multi-scale', action='store_true', help='random image sizes per batch 320 - 608')
    parser.add_argument('--img-size', type=int, default=32 * 13, help='pixels')
    parser.add_argument('--weights-path', type=str, default='weights', help='path to store weights')
    parser.add_argument('--resume', action='store_true', help='resume training flag')
    parser.add_argument('--freeze', action='store_false', help='freeze darknet53.conv.74 layers')
    parser.add_argument('--freeze-for-epochs', type=int, default=100, help='freeze darknet53.conv.74 layers for specified epochs')
    parser.add_argument('--var', type=float, default=0, help='optional test variable')
    opt = parser.parse_args()
    print(opt, end='\n\n')

    init_seeds()

    torch.cuda.empty_cache()
    train(
        opt.cfg,
        opt.data_config,
        img_size=opt.img_size,
        resume=opt.resume,
        epochs=opt.epochs,
        batch_size=opt.batch_size,
        accumulated_batches=opt.accumulated_batches,
        weights_path=opt.weights_path,
        multi_scale=opt.multi_scale,
        freeze_backbone=opt.freeze,
        var=opt.var,
        lr=opt.learning_rate,
        freeze_for=opt.freeze_for_epochs
    )
