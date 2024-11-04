import os
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
from collections import Counter
import re


# 构建词典的函数
def build_vocab_from_captions(csv_file, min_freq=1):
    data = pd.read_csv(csv_file)
    captions = data['caption']  # 获取所有报告

    word_counter = Counter()

    # 遍历每个报告，分词并统计词频
    for caption in captions:
        # 将所有单词转换为小写，并使用正则去掉标点符号
        tokens = re.findall(r'\b\w+\b', caption.lower())
        word_counter.update(tokens)

    # 创建词典，min_freq 表示至少出现 min_freq 次的单词才会进入词典
    word_to_idx = {word: idx + 1 for idx, (word, freq) in enumerate(word_counter.items()) if freq >= min_freq}

    # 加入特殊的填充符号 (PAD) 和未知词 (UNK)
    word_to_idx['<PAD>'] = 0
    word_to_idx['<UNK>'] = len(word_to_idx)

    return word_to_idx


# 分词器，传入词典将报告转换为词 ID
def tokenize(caption, word_to_idx, max_caption_len=20):
    tokens = caption.lower().split()  # 分词
    token_ids = [word_to_idx.get(token, word_to_idx['<UNK>']) for token in tokens]  # 未知词使用 <UNK> ID
    if len(token_ids) < max_caption_len:
        token_ids += [word_to_idx['<PAD>']] * (max_caption_len - len(token_ids))  # 填充
    return torch.tensor(token_ids[:max_caption_len])  # 返回固定长度的 Tensor


# 数据集定义
class IUXRayDataset(Dataset):
    def __init__(self, csv_file, image_base_dir, word_to_idx, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_base_dir = image_base_dir
        self.word_to_idx = word_to_idx  # 保存词典
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.loc[idx, 'dir']
        caption = self.data.loc[idx, 'caption']

        # 去掉前缀 "F1 "、"F2 " 等
        img_name = img_name.replace("F1 ", "").replace("F2 ", "").replace("F3 ", "").replace("F4 ", "")
        img_path = os.path.join(self.image_base_dir, img_name + ".png")

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image at {img_path}: {e}")
            return None, None

        if self.transform:
            image = self.transform(image)

        # 使用词典将报告转换为词 ID 序列
        caption_tensor = tokenize(caption, self.word_to_idx)

        return image, caption_tensor


# 假设词典已经生成
csv_file = 'E:/jsjshij/8222/IUxRay.csv'
image_base_dir = 'E:/jsjshij/8222/images'

# 使用生成的词典 word_to_idx
word_to_idx = build_vocab_from_captions(csv_file)

# 图像变换
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载数据集，传递词典 word_to_idx
dataset = IUXRayDataset(csv_file=csv_file, image_base_dir=image_base_dir, word_to_idx=word_to_idx, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 验证数据加载
for images, captions in dataloader:
    print("Image batch shape:", images.shape)
    print("Captions batch shape:", captions.shape)  # 打印 captions 的形状
    break  # 打印一个批次的数据
