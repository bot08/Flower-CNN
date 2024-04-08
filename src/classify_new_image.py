# .venv\Scripts\activate
# python E:\AI_a\CNN-10\src\classify_new_image.py

import torch
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from PIL import Image
from cnn_main import CNNet
from pathlib import Path

model = CNNet(8) # NUM CLASS
checkpoint = torch.load(Path('E:\\AI_a\\18-8.model')) # Pretrained model load
model.load_state_dict(checkpoint)
trans = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
])

model.eval()

image = Image.open(Path('E:\\AI_a\\iris-test1.jpg'))
input_loader = DataLoader(dataset=image, batch_size=1, shuffle=False, num_workers=5)
input_loader = trans(image)
input_loader = input_loader.view(1, 3, 32,32)

output = model(input_loader)

prediction = output.data.numpy().argmax()
print(output.data.numpy())

if (prediction == 0):
    print ('bellflower')
if (prediction == 1):
    print ('daisy')
if (prediction == 2):
    print ('dandelion')
if (prediction == 3):
    print ('iris')
if (prediction == 4):
    print ('rose')
if (prediction == 5):
    print ('sunflower')
if (prediction == 6):
    print ('tulip')
if (prediction == 7):
    print ('water_lily')