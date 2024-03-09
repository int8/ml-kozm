import os
import numpy as np
import torch
from PIL import Image
import xml.etree.ElementTree as ET
from torchvision import transforms

from torch.utils.tensorboard import SummaryWriter


class PlasmodiumDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        image_files = sorted(
            [img for img in os.listdir(root) if img.endswith(".jpg")])
        self.imgs = []
        for img_file in image_files:
            xml_file = os.path.join(self.root, img_file.replace(".jpg", ".xml"))
            tree = ET.parse(xml_file)
            if tree.findall(
                    'object'):  # check if 'object' tag exists in xml file
                self.imgs.append(img_file)

    def __getitem__(self, idx):
        # load images and labels
        img_path = os.path.join(self.root, self.imgs[idx])
        label_path = os.path.join(self.root,
                                  self.imgs[idx].replace(".jpg", ".xml"))
        img = Image.open(img_path).convert("RGB")

        # read xml file
        tree = ET.parse(label_path)
        boxes = []
        masks = []

        for obj in tree.findall('object'):
            mask = np.zeros((img.size[1], img.size[0]), dtype=np.uint8)
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            boxes.append([xmin, ymin, xmax, ymax])
            mask[int(ymin):int(ymax), int(xmin):int(xmax)] = 1
            masks.append(mask)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((len(boxes),), dtype=torch.int64)
        masks = np.array(masks)
        masks = torch.as_tensor(masks, dtype=torch.uint8)  # for a binary mask

        image_id = torch.tensor([idx])
        if len(boxes):
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        else:
            area = torch.Tensor([])

        # suppose all instances are not crowd
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        if self.transforms is not None:
            img = self.transforms(img)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["masks"] = masks

        return img, target

    def __len__(self):
        return len(self.imgs)


my_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = PlasmodiumDataset(
    "/plasmodium-phonecamera", transforms=my_transforms
)

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

# Define the dataloader
data_loader = torch.utils.data.DataLoader(dataset, batch_size=8,
                                          shuffle=True, num_workers=2,
                                          collate_fn=lambda x: tuple(zip(*x)))



###### cell 2

# Load a model pre-trained on COCO
model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(pretrained=True)


# Replace the classifier with a new one (number of classes = 2: background and plasmodium)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
hidden_layer = 256
model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, 2)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Move model to the device
model.to(device)

# Parameters
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr=0.005) #, lr=0.005,
                            #momentum=0.9, weight_decay=0.0005)
# Training loop
num_epochs = 50

writer = SummaryWriter()
i = 0
for epoch in range(num_epochs):
    model.train()
    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        writer.add_scalar('training loss', losses,i)
        i+=1
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

# Save the trained model
torch.save(model.state_dict(), 'model.pth')
writer.close()



# eval / test

import numpy as np
import cv2
import torchvision.transforms as T
from PIL import Image
import torch
import torchvision
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from matplotlib import pyplot as plt

def get_prediction(img_path, threshold):
    img = Image.open(img_path).convert("RGB")
    img_t = transform(img)
    model.eval()
    with torch.no_grad():
        prediction = model([img_t.to(device)])

    # Keep only prediction with score > threshold
    prediction = [{k: v[prediction[0]['scores'] > threshold] for k, v in prediction[0].items()}]
    return prediction


def draw_boxes(img_path, prediction):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for i in range(prediction[0]["boxes"].shape[0]):
        xmin, ymin, xmax, ymax = map(int, prediction[0]["boxes"][i])
        label = int(prediction[0]["labels"][i])
        score = float(prediction[0]["scores"][i])
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color=(0, 255, 0), thickness=2)
        cv2.putText(img, str(label) + ": " + str(round(score, 2)), (xmin, ymin-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, lineType=cv2.LINE_AA)
    plt.figure(figsize=(10,10))
    plt.imshow(img)
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()


# Load your trained model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
hidden_layer = 256
model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, 2)
model.load_state_dict(torch.load('model.pth'))
model.to(device)

transform = T.Compose([T.ToTensor()])

# Get image path
img_path = "/plasmodium-phonecamera/plasmodium-phone-1079.jpg"  # Put your test image path here

# Get predictions
preds = get_prediction(img_path, threshold=0.1)

# Draw boxes
draw_boxes(img_path, preds)