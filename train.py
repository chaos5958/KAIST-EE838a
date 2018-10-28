import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import imageio
from model import UNet
from option import opt
from dataset import LdrHdrDataset
from utility import getLogger
from datetime import datetime

def custom_loss(predict, target, alpha):
    predict = torch.mul(alpha, predict)
    target = torch.mul(alpha, torch.log(target + 0.01))

    tensor_shape = target.size()
    num_pixels = tensor_shape[0] * tensor_shape[1] * tensor_shape[2] * tensor_shape[3]

    loss = torch.mean(((predict - target) ** 2))

    return loss

def BlendMap(image):
    max_input, _ = torch.max(image, dim=1)
    alpha = torch.clamp(max_input - 0.95, min=0) / 0.05
    alpha = torch.stack((alpha, alpha, alpha), dim=1)

    return alpha

def train():
    device = torch.device("cuda" if opt.use_cuda else "cpu")

    model = UNet().to(device)
    train_set = LdrHdrDataset('train/HDR', True, opt.num_batch_train)
    train_loader = DataLoader(train_set, batch_size=opt.num_batch_train, shuffle=True, num_workers=4)

    valid_set = LdrHdrDataset('val/HDR', False, opt.num_batch_test)
    valid_loader = DataLoader(valid_set, batch_size=opt.num_batch_test, shuffle=False, num_workers=1)

    loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    log_name = datetime.now().strftime('valid_result_%H_%M_%d_%m_%Y.log')
    valid_logger = getLogger(opt.result_dir, log_name)
    valid_logger.info('Epoch\tIndex\tLoss')

    for epoch in range(opt.epoch):
        #Train
        model = model.train()
        for iteration, batch in enumerate(train_loader):
            input, target = batch[0].to(device), batch[1].to(device)

            alpha = BlendMap(input)
            predict = model(input, alpha)

            if opt.loss_fn == "log_mse":
                output = custom_loss(predict, target, alpha)
            elif opt.loss_fn == "mse":
                output = loss(predict, target)
            print('{} epoch - {}/{} mini-batch: {}'.format(epoch+1, iteration+1, int(len(train_set) / opt.num_batch_train), output))

            optimizer.zero_grad()
            output.backward()
            optimizer.step()

        #Save a model
        torch.save(model.state_dict(), '{}/{}'.format(opt.model_dir, 'epoch{}.pth'.format(epoch)))

        #Validation
        model = model.eval()
        with torch.no_grad():
            for iteration, batch in enumerate(valid_loader):
                input, target = batch[0].to(device), batch[1].to(device)

                alpha = BlendMap(input)
                predict = model(input, alpha)

                output = custom_loss(predict, target, alpha)
                print('{} epoch ({} validation): {}'.format(epoch+1, iteration+1, output))
                valid_logger.info('{}\t{}\t{}'.format(epoch+1, iteration+1, output))

                #Transform to numpy images
                input = input.data[0].permute(1,2,0).cpu().numpy()
                predict = predict.data[0].permute(1,2,0).cpu().numpy()
                target = target.data[0].permute(1,2,0).cpu().numpy()

                input *= 255
                input = input.astype(np.uint8)

                #Save images
                imageio.imwrite('{}/epoch{}-input{}.png'.format(opt.result_dir, epoch, iteration), input)
                imageio.imwrite('{}/epoch{}-predict{}.hdr'.format(opt.result_dir, epoch, iteration), predict)
                imageio.imwrite('{}/epoch{}-target{}.hdr'.format(opt.result_dir, epoch, iteration), target)

if __name__ == "__main__":
    train()
