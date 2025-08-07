import os
import math
import argparse
import json
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import transforms


from SCCNet import sccnet50 as create_model
from utils import train_one_epoch, evaluate, write_log


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if os.path.exists("./weights") is False:
        os.mkdir("./weights")
    tb_writer = SummaryWriter(log_dir="./weights")

    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }


    train_dataset = torchvision.datasets.ImageNet(root=args.data_path,split='train',
                                               transform=data_transform["train"])

    val_dataset = torchvision.datasets.ImageNet(root=args.data_path,split='val',
                                               transform=data_transform["val"])


    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process in {}'.format(nw,device))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True,
                                               num_workers=nw)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True,
                                             num_workers=nw)


    model = create_model(num_classes=args.num_classes).to(device)

    mx_acc = 0.60

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file not found"
        weights_dict = torch.load(args.weights, map_location=device)
        print(model.load_state_dict(weights_dict, strict=False))
        _, mx_acc, _ = evaluate(model=model,
                             test_loader=val_loader,
                             device=device,
                             epoch= 'for weight validity verification')
    if args.freeze_layers:
        for name, para in model.named_parameters():
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5E-5)
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                train_loader=train_loader,
                                                device=device,
                                                epoch=epoch)

        scheduler.step()

        # validate
        val_loss, val_acc1, val_acc5 = evaluate(model=model,
                                     test_loader=val_loader,
                                     device=device,
                                     epoch=epoch)


        tags = ["train_loss", "train_acc", "val_loss", "top1_acc","top5_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc1, epoch)
        tb_writer.add_scalar(tags[4], val_acc5, epoch)
        tb_writer.add_scalar(tags[5], optimizer.param_groups[0]["lr"], epoch)
        if val_acc1 > mx_acc:
            mx_acc = val_acc1
            print("New best weight appears!")
            torch.save(model.state_dict(), "./weights/{}-best.pth".format(args.model_name))
        if epoch % 2 == 1:
            torch.save(model.state_dict(), "./weights/{}-latest.pth".format(args.model_name))
        write_log("./log/{} training.log".format(args.model_name),tags, epoch, train_loss, train_acc, val_loss, val_acc1,val_acc5, optimizer.param_groups[0]["lr"])



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-classes', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--lrf', type=float, default=0.001)

    parser.add_argument('--data-path', type=str,
                        default="./data/imagenet1k")
    parser.add_argument('--model-name', default='model', help='create model name')

    parser.add_argument('--weights', type=str, default="",
                        help='initial weights path')

    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
