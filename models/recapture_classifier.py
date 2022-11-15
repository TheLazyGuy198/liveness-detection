import torch
from torch import nn
import os
import timm
import albumentations as A
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2

from utils.misc import parse_classify_model_name


class RecaptureClassifier(nn.Module):

    def __init__(self, arch='tf_efficientnetv2_b1', n_classes=2, pretrained=False):
        super().__init__()
        self.model = timm.create_model(arch, pretrained=pretrained)
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(n_features, n_classes)

    def forward(self, x):
        x = self.model(x)
        return x


class RecaptureDetector:

    def __init__(self, device_id, weight_path):
        model_name = os.path.basename(weight_path)
        h_input, w_input, arch = parse_classify_model_name(model_name)
        self.model = RecaptureClassifier(arch=arch, pretrained=True)
        self.device = torch.device("cuda:{}".format(device_id) if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.load_weight(weight_path)

    def load_weight(self, weight_path):
        model_name = os.path.basename(weight_path)
        h_input, w_input, arch = parse_classify_model_name(model_name)
        state_dict = torch.load(weight_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.transforms = A.Compose([
            A.Resize(height=h_input, width=w_input, p=1.0),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ], p=1.0)

    def predict(self, image):
        image = self.transforms(image=image)['image']
        image = image.unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            preds = self.model.forward(image)
            logits = F.softmax(preds).cpu().numpy()
        return logits
