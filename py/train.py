#-*-coding:utf-8-*-
"""
    This is a template for torch main.py
"""
import os
import torch
import shutil
import argparse
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from torch.autograd import Variable as Var
from torchvision.utils import save_image
from costVolume import CostVolume

from torch import optim
from torch import nn
from torchvision import transforms

tf = transforms.ToTensor()
normed_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

def trainLoader():
    data = []
    for root, dirs, _ in os.walk("../data/MiddEval3/trainingQ/"):
        for _dir in dirs:
            path_prefix = os.path.join(root, _dir)
            gt_path = "%s/%s"%(path_prefix, "gt_disp.bin")
            left_path = "%s/%s"%(path_prefix, "im0.png")
            right_path = "%s/%s"%(path_prefix, "im1.png")
            left_raw = plt.imread(left_path)
            right_raw = plt.imread(right_path)
            gt_raw = np.fromfile(gt_path, dtype=np.float64)
            w, h, _ = gt_raw[:3].astype(int)
            gt_img = gt_raw[3:].reshape(h, w, 1)
            data.append((normed_tf(left_raw), normed_tf(right_raw), tf(gt_img.astype(np.float32))))
    return data

def testLoader():
    data = []
    for root, dirs, _ in os.walk("../data/MiddEval3/testQ/"):
        for _dir in dirs:
            path_prefix = os.path.join(root, _dir)
            left_path = "%s/%s"%(path_prefix, "im0.png")
            right_path = "%s/%s"%(path_prefix, "im1.png")
            left_raw = plt.imread(left_path)
            right_raw = plt.imread(right_path)
            data.append((normed_tf(left_raw), normed_tf(right_raw)))
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type = int, default = 50, help = "Training lasts for . epochs")
    parser.add_argument("--max_iter", type = int, default = 3, help = "max iteration number")
    parser.add_argument("--max_disp", type = int, default = 200, help = "Fixed and unified maximum disparity")
    parser.add_argument("--res_num", type = int, default = 3, help = "Residual block number in encoder")
    parser.add_argument("--eval_time", type = int, default = 10, help = "Evaluate every <test_time> times")
    parser.add_argument("-d", "--del_dir", action = "store_true", help = "Delete dir ./logs and start new tensorboard records")
    parser.add_argument("-l", "--load", action = "store_true", help = "Load from saved model or check points")
    parser.add_argument("-c", "--cuda", default = False, action = "store_true", help = "Use CUDA to speed up training")
    args = parser.parse_args()
    epochs = args.epochs
    del_dir = args.del_dir
    use_cuda = args.cuda
    max_iter = args.max_iter

    tf = transforms.ToTensor()
    vol = CostVolume(args.res_num, args.max_disp, use_cuda)
    if use_cuda and torch.cuda.is_available():
        vol = vol.cuda()
    else:
        torch.cuda.empty_cache()
        print("Not using CUDA.")
    logdir = '../logs/'
    if os.path.exists(logdir) and del_dir:
        shutil.rmtree(logdir)
    time_stamp = "{0:%Y-%m-%d/%H-%M-%S}-epoch{1}/".format(datetime.now(), epochs)
    # writer = SummaryWriter(log_dir = logdir+time_stamp)
    train_set = trainLoader()
    test_set = testLoader()
    loss_func = nn.MSELoss()
    opt = optim.Adam(vol.parameters(), lr = 1e-3)
    if args.load:
        save = torch.load("../chpt/eval_59.pt")
        save_model = save['model']
        model_dict = vol.state_dict()
        state_dict = {k:v for k, v in save_model.items() if k in model_dict}
        model_dict.update(state_dict)
        vol.load_state_dict(model_dict) 
    # with torch.no_grad():
    #     for cnt, (l_img, r_img) in enumerate(test_set):
    #         l_img = l_img.unsqueeze(dim = 0)
    #         r_img = r_img.unsqueeze(dim = 0)
    #         if use_cuda:
    #             l_img = l_img.cuda()
    #             r_img = r_img.cuda()
    #         gen = vol.forward(l_img, r_img)
    #         print(torch.max(gen))
    #         gen /= torch.max(gen)
    #         save_image(gen.detach().clamp_(0, 1), "../imgs/G_%d_%d.jpg"%(6 + 1, cnt + 1), nrows = 1)
    # vol = vol.train()
    eval_cnt = 0
    print("Starting to train.")
    for i in range(epochs):
        for j, (l_img, r_img, gt) in enumerate(train_set):
            l_img = l_img.unsqueeze(dim = 0)
            r_img = r_img.unsqueeze(dim = 0)
            if use_cuda:
                l_img = l_img.cuda()
                r_img = r_img.cuda()
                gt = gt.cuda()
            for k in range(max_iter):
                opt.zero_grad()
                out = vol.forward(l_img, r_img)
                loss = loss_func(out, gt)
                loss.backward()
                opt.step()
                print("Epoch: %3d / %3d\t Image: %2d / %2d\t Iter %4d / %4d\t train loss: %.4f\t"%(
                    i, epochs, j, len(train_set), k, max_iter, loss.item()
                ))
        if i % args.eval_time: continue
        vol = vol.eval()
        torch.save({'model': vol.state_dict(), 'optimizer': opt.state_dict()}, "../chpt/eval_%d.pt"%(eval_cnt + 40))
        with torch.no_grad():
            for cnt, (l_img, r_img) in enumerate(test_set):
                l_img = l_img.unsqueeze(dim = 0)
                r_img = r_img.unsqueeze(dim = 0)
                if use_cuda:
                    l_img = l_img.cuda()
                    r_img = r_img.cuda()
                gen = vol.forward(l_img, r_img)
                print(torch.max(gen))
                gen /= torch.max(gen)
                save_image(gen.detach().clamp_(0, 1), "../imgs/D_%d_%d.jpg"%(eval_cnt + 40, cnt + 1), nrows = 1)
        vol = vol.train()
        eval_cnt += 1
    # writer.close()
    print("Output completed.")
    torch.save({'model': vol.state_dict(), 'optimizer': opt.state_dict()}, "../model/stereo2.pth")
    