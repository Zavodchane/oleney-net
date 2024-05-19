import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch
from PIL import Image

from windows.database import save_imgs_metadata

labelsToClassDict = {
    0 : "Кабарга",
    1 : "Косуля",
    2 : "Олень"
}

class CustomDataset(Dataset):
  def __init__(self, files : list, transform=None):
    self.transform = transform
    self.data = files

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    try:
        tf = transforms.Compose([
            transforms.Resize((420, 420)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        data = self.data[idx]
        img = Image.open(data).convert('RGB')
        img = tf(img)
        return img

    except Exception as e:
        print(e)


def classify(imgsPath: list, model, device):
    dataset = CustomDataset(imgsPath)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=False)
    model.load_state_dict(torch.load("dino/dinov2_weights.pth", map_location=device))
    model.eval()

    labels = []
    for batch in dataloader:
        outputs = model.dinov2(batch.to(device))
        outputs = model.classifier(outputs.pooler_output)
        _, preds = torch.max(outputs, 1)
        labels.extend(preds.cpu().tolist())

    classes = labelsToClass(labels)
    save_imgs_metadata(imgs_path=imgsPath, classes=classes)

    print(classes)


def labelsToClass(labels : list):
    classList = list(map(lambda x: labelsToClassDict[x], labels))
    return classList