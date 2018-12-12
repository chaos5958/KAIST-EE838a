import numpy as np
import time, logging, os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from model import MultiNet
from option import opt
from dataset import DeblurDataset
from datetime import datetime

def getLogger(save_dir, save_name):
    Logger = logging.getLogger(save_name)
    Logger.setLevel(logging.INFO)
    Logger.propagate = False

    filePath = os.path.join(save_dir, save_name)
    if os.path.exists(filePath):
        os.remove(filePath)

    fileHandler = logging.FileHandler(filePath)
    logFormatter = logging.Formatter('%(message)s')
    fileHandler.setFormatter(logFormatter)
    Logger.addHandler(fileHandler)

    return Logger

def train():
    #Create a model
    device = torch.device("cuda" if opt.use_cuda else "cpu")
    model = MultiNet().to(device)

    #Create dataset
    train_set = DeblurDataset(opt.train_dir, True)
    train_loader = DataLoader(train_set, batch_size=opt.num_batch_train, shuffle=True, num_workers=4)

    valid_set = DeblurDataset(opt.test_dir, False)
    valid_loader = DataLoader(valid_set, batch_size=opt.num_batch_test, shuffle=False, num_workers=1)

    loss = torch.nn.MSELoss(reduction='elementwise_mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    log_name = datetime.now().strftime('valid_result_%H_%M_%d_%m_%Y.log')
    valid_logger = getLogger(opt.result_dir, log_name)
    valid_logger.info('Epoch\tLoss')

    for epoch in range(opt.epoch):
        #Train
        epoch_train_start = time.time()
        model = model.train()
        for iteration, batch in enumerate(train_loader):
            x4_input = batch[0].to(device)
            x2_input = batch[1].to(device)
            x1_input = batch[2].to(device)
            x4_target = batch[3].to(device)
            x2_target = batch[4].to(device)
            x1_target = batch[5].to(device)

            #Predict
            x4_output, x2_output, x1_output = model(x4_input, x2_input, x1_input)

            #Custom loss
            x4_loss = loss(x4_output, x4_target)
            x2_loss = loss(x2_output, x2_target)
            x1_loss = loss(x1_output, x1_target)
            total_loss = (x4_loss + x2_loss + x1_loss) / 6

            print('{} epoch - {}/{} mini-batch: {}'.format(epoch+1, iteration+1, int(len(train_set) / opt.num_batch_train), total_loss))

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        epoch_train_end = time.time()
        print('{} epoch train: {} seconds elapsed'.format(epoch+1, epoch_train_end - epoch_train_start))

        #Save a model
        torch.save(model.state_dict(), '{}/{}'.format(opt.model_dir, 'epoch{}.pth'.format(epoch)))

        #Validation
        epoch_valid_start = time.time()
        model = model.eval()
        with torch.no_grad():
            count = 0
            for iteration, batch in enumerate(valid_loader):
                x4_input = batch[0].to(device)
                x2_input = batch[1].to(device)
                x1_input = batch[2].to(device)
                x4_target = batch[3]
                x2_target = batch[4]
                x1_target = batch[5]

                #Predict
                x4_output, x2_output, x1_output = model(x4_input, x2_input, x1_input)

                #Clamp values to (0-1)
                x4_output = torch.clamp(x4_output, min=0, max=1)
                x2_output = torch.clamp(x2_output, min=0, max=1)
                x1_output = torch.clamp(x1_output, min=0, max=1)

                #Transform to numpy and save images
                if epoch == 0:
                    x4_input_img = transforms.ToPILImage()(x4_input.data[0].to('cpu'))
                    x4_input_img = transforms.ToPILImage()(x4_input.data[0].to('cpu'))
                    x2_input_img = transforms.ToPILImage()(x2_input.data[0].to('cpu'))
                    x1_input_img = transforms.ToPILImage()(x1_input.data[0].to('cpu'))
                    x4_target_img = transforms.ToPILImage()(x4_target.data[0])
                    x2_target_img = transforms.ToPILImage()(x2_target.data[0])
                    x1_target_img = transforms.ToPILImage()(x1_target.data[0])

                    x4_input_img.save('{}/x4-input{}.png'.format(opt.result_dir, epoch, count), 'png')
                    x2_input_img.save('{}/x2-input{}.png'.format(opt.result_dir, epoch, count), 'png')
                    x1_input_img.save('{}/x1-input{}.png'.format(opt.result_dir, epoch, count), 'png')
                    x4_target_img.save('{}/x4-target{}.png'.format(opt.result_dir, epoch, count), 'png')
                    x2_target_img.save('{}/x2-target{}.png'.format(opt.result_dir, epoch, count), 'png')
                    x1_target_img.save('{}/x1-target{}.png'.format(opt.result_dir, epoch, count), 'png')

                x4_output_img = transforms.ToPILImage()(x4_output.data[0].to('cpu'))
                x2_output_img = transforms.ToPILImage()(x2_output.data[0].to('cpu'))
                x1_output_img = transforms.ToPILImage()(x1_output.data[0].to('cpu'))
                x4_output_img.save('{}/epoch{}-x4-output{}.png'.format(opt.result_dir, epoch, count), 'png')
                x2_output_img.save('{}/epoch{}-x2-output{}.png'.format(opt.result_dir, epoch, count), 'png')
                x1_output_img.save('{}/epoch{}-x1-output{}.png'.format(opt.result_dir, epoch, count), 'png')

                count = count + 1

        epoch_valid_end = time.time()
        print('{} epoch valid: {} seconds elapsed'.format(epoch+1, epoch_valid_end - epoch_valid_start))

if __name__ == "__main__":
    train()
