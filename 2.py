import os
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class IUXRayDataset(Dataset):
    def __init__(self, csv_file, image_base_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_base_dir = image_base_dir  # 图片的根目录
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.loc[idx, 'dir']
        caption = self.data.loc[idx, 'caption']

        # 逐步去掉前缀 "F1 "、"F2 "、"F3 "、"F4 "
        img_name = img_name.replace("F1 ", "").replace("F2 ", "").replace("F3 ", "").replace("F4 ", "")

        # 拼接图片路径和添加扩展名（假设是 .png）
        img_path = os.path.join(self.image_base_dir, img_name + ".png")

        # 调试信息：打印图片路径
        print(f"Loading image from: {img_path}")

        try:
            # 打开图像
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image at {img_path}: {e}")
            return None, None  # 如果加载失败，返回 None

        if self.transform:
            image = self.transform(image)

        return image, caption

# 图像变换 (数据增强和归一化)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载数据集
csv_file = 'E:/jsjshij/8222/IUxRay.csv'
image_base_dir = 'E:/jsjshij/8222/images'  # 图像的根目录
dataset = IUXRayDataset(csv_file=csv_file, image_base_dir=image_base_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 验证数据加载
for images, captions in dataloader:
    # 跳过加载失败的图像
    if images is None or captions is None:
        continue

    print("Image batch shape:", images.shape)
    print("Captions:", captions[:5])  # 打印前5个caption
    break  # 打印一个批次的数据
