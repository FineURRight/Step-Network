import time
from options.train_options import TrainOptions
import data as Dataset
from model.networks.keypoints_generator import KPModel
import torch.nn as nn
import torch


if __name__ == '__main__':
    # get training options
    opt = TrainOptions().parse()
    # create a dataset
    dataset = Dataset.create_dataloader(opt)

    device = torch.device("cuda:0")
    model = KPModel()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(
        0.9, 0.999), eps=1e-08, weight_decay=0.1)
    cr1 = nn.CrossEntropyLoss()

    for epoch in range(20):
        total_loss=0
        n=0
        for i, data in enumerate(dataset):
            model.keypoint_input(data['BP1_cor'])
            optimizer.zero_grad()
            output, loss, _ = model()
            # loss= cr(output[tm],gt[tm])
            total_loss+=loss.data
            print(str(i)+'/'+str(len(dataset)))
            print("now loss:")
            print(loss.data)
            loss.backward()
            optimizer.step()
            n=i
        if (epoch+1)%5==0:
            torch.save(model.state_dict(), './pretrained_model/k_generator'+str(epoch+1)+'.pth')
        print("epoch"+str(epoch+1)+"loss:")
        print(total_loss/n)
            
