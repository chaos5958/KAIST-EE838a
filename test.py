import torch
from torch.utils.data import Dataset, DataLoader
import imageio
from model import UNet
from option import opt
from dataset import LdrHdrDataset
from utility import getLogger
from datetime import datetime
import os
import numpy as np

def custom_loss(predict, target, alpha):
    predict = torch.mul(alpha, predict)
    target = torch.mul(alpha, torch.log(target + 0.01))

    loss = torch.mean(((predict - target) ** 2))

    return loss

def BlendMap(image):
    max_input, _ = torch.max(image, dim=1)
    alpha = torch.clamp(max_input - 0.95, min=0) / 0.05
    alpha = torch.stack((alpha, alpha, alpha), dim=1)

    return alpha

def test():
    device = torch.device("cuda" if opt.use_cuda else "cpu")
    model = UNet().to(device)

    valid_set = LdrHdrDataset(opt.test_dir, False, opt.num_batch_test)
    valid_loader = DataLoader(valid_set, batch_size=opt.num_batch_test, shuffle=False, num_workers=1)

    #Model loading
    model.load_state_dict(torch.load(os.path.join(opt.model_dir, opt.model_name)))
    model = model.eval()

    #Test
    with torch.no_grad():
        for iteration, batch in enumerate(valid_loader):
            input, target = batch[0].to(device), batch[1].to(device)
            normalized_value = batch[2].numpy()

            alpha = BlendMap(input)
            predict = model(input, alpha)

            output = custom_loss(predict, target, alpha)

            #Transform to numpy images
            input = input.data[0].permute(1,2,0).cpu().numpy()
            predict = predict.data[0].permute(1,2,0).cpu().numpy()
            target = target.data[0].permute(1,2,0).cpu().numpy()

            input *= 255
            input = input.astype(np.uint8)
            predict = predict * normalized_value
            target = target * normalized_value

            #Save images
            imageio.imwrite('{}/input{}.png'.format(opt.result_dir, iteration), input)
            imageio.imwrite('{}/predict{}.hdr'.format(opt.result_dir, iteration), predict)
            imageio.imwrite('{}/target{}.hdr'.format(opt.result_dir, iteration), target)

if __name__ == "__main__":
    test()
