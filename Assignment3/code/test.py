import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import numpy as np
from net import Net
import dataload
from torchvision import transforms
import torch.nn.functional as F
import utils
import cv2
import os
import matplotlib
import matplotlib.image as img
def test(test_loader, test_net, threshold, device, batch_size) :
    test_net.eval()

    total_num = 0

    Rel_error_all = 0.0
    lg10_error_all = 0.0
    result_path = 'test_result'
    if not os.path.exists(result_path) :
        os.mkdir(result_path)
    with torch.no_grad() :
        for i, test_sample in enumerate(test_loader) :
            image = test_sample['image'].to(device)
            depth = test_sample['depth'].to(device)
            name = test_sample['name'][0]
            output = test_net(image)
            output = F.upsample(output, size=[228,304], mode='bilinear')
            output_img = output.view(228,304).data.cpu().numpy()
            output_img = np.uint16(output_img * 1000.)
            #matplotlib.image.imsave('./{}/visual_img_{}.png'.format(result_path, name), output_img)
            cv2.imwrite('./{}/img_{}.png'.format(result_path, name), output_img)
            total_num += batch_size

            rel_error, lg10_error = utils.evaluate_error(output, depth)
            Rel_error_all += rel_error
            lg10_error_all += lg10_error
    print("Average Rel_error : {}, Average lg10_error : {}".format(Rel_error_all/float(total_num),lg10_error_all/float(total_num)))


if __name__ == "__main__" :
    #parameters
    batch_size = 1
    threshold = 0.25
    

    test_net = Net()
    test_net.load_state_dict(torch.load('./train_net.pth'))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #print(device)
    test_net.to(device)


    test_loader = dataload.load_data('./nyuv2/test/', batch_size, transform=transforms.Resize((228,304)))
    test(test_loader, test_net, threshold, device, batch_size)