from torchsummary import summary
import torchvision
import torch
from model.model_my import background_resnet

def model_info(model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    backbone = model.to(device)
    summary(model, (1, 40, 100))

mymodel = background_resnet(512,340)
#print(mymodel)
model_info(mymodel)

