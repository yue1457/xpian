import os
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from collections import Counter
import re
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import threading

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

# 图像编码器 (使用预训练的 ResNet50)
class CNN_Encoder(nn.Module):
    def __init__(self, embed_size):
        super(CNN_Encoder, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)  # 使用最新的 weights 参数
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.fc = nn.Linear(resnet.fc.in_features, embed_size)
        self.relu = nn.ReLU()

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.fc(features)
        features = self.relu(features)
        return features

# Transformer 解码器
class Transformer_Decoder(nn.Module):
    def __init__(self, embed_size, num_heads, num_layers, vocab_size, max_len):
        super(Transformer_Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_len, embed_size))  # 位置编码
        decoder_layer = nn.TransformerDecoderLayer(embed_size, num_heads)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, features, captions, padding_mask=None):
        embeddings = self.embedding(captions) + self.positional_encoding[:, :captions.size(1), :]
        embeddings = embeddings.transpose(0, 1)  # Transformer expects [seq_len, batch_size, embed_size]
        features = features.unsqueeze(1).transpose(0, 1)  # Adjust to [1, batch_size, embed_size]

        # 传递 padding_mask 作为 tgt_key_padding_mask
        output = self.transformer_decoder(embeddings, features, tgt_key_padding_mask=padding_mask)
        output = output.transpose(0, 1)  # Convert back to [batch_size, seq_len, embed_size]
        output = self.fc(output)
        return output

# 训练函数
# 训练函数

# 训练函数
def train_model(encoder, decoder, dataloader, criterion, optimizer, num_epochs, save_interval, word_to_idx, max_len=20, save_path="models/best_model.pth"):
    encoder.train()
    decoder.train()
    scaler = GradScaler()

    best_loss = float('inf')

    # 检查并创建 models 文件夹
    if not os.path.exists("models"):
        os.makedirs("models")

    for epoch in range(num_epochs):
        total_loss = 0

        for images, captions in tqdm(dataloader, desc=f"Epoch [{epoch + 1}/{num_epochs}]"):
            images = images.cuda()
            captions = captions.cuda()

            optimizer.zero_grad()

            padding_mask = (captions == word_to_idx['<PAD>']).cuda()

            with autocast():
                features = encoder(images)
                outputs = decoder(features, captions, padding_mask=padding_mask)
                loss = criterion(outputs.view(-1, outputs.size(-1)), captions.view(-1))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, save_path)
            print(f"Model saved with loss: {best_loss:.4f}")

        if (epoch + 1) % save_interval == 0:
            model_save_path = f"models/model_epoch_{epoch + 1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, model_save_path)
            print(f"Model checkpoint saved at {model_save_path}")

# GUI 相关函数
def start_training():
    try:
        num_epochs = int(epoch_entry.get())
        save_interval = int(save_interval_entry.get())

        if num_epochs <= 0 or save_interval <= 0:
            messagebox.showerror("错误", "请输入正整数！")
            return

        # 将模型训练放到一个新的线程中
        train_thread = threading.Thread(target=run_training, args=(num_epochs, save_interval))
        train_thread.start()

    except ValueError:
        messagebox.showerror("错误", "请输入有效的数字！")

def run_training(num_epochs, save_interval):
    try:
        # 假设词典已经生成
        csv_file = 'E:/jsjshij/8222/IUxRay.csv'
        image_base_dir = 'E:/jsjshij/8222/images'

        word_to_idx = build_vocab_from_captions(csv_file)
        if not isinstance(word_to_idx, dict):
            messagebox.showerror("错误", "词典生成失败，请检查词典生成函数。")
            return

        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        dataset = IUXRayDataset(csv_file=csv_file, image_base_dir=image_base_dir, word_to_idx=word_to_idx, transform=transform)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        embed_size = 128
        num_heads = 4
        num_layers = 6
        vocab_size = len(word_to_idx) + 1
        max_len = 20

        encoder = CNN_Encoder(embed_size).cuda()
        decoder = Transformer_Decoder(embed_size, num_heads, num_layers, vocab_size, max_len).cuda()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001)

        # 训练模型
        train_model(encoder, decoder, dataloader, criterion, optimizer, num_epochs, save_interval, word_to_idx, max_len, save_path="models/best_model.pth")
        messagebox.showinfo("完成", "训练完成！")

    except Exception as e:
        messagebox.showerror("错误", f"训练过程中出现错误: {str(e)}")
# 创建 GUI 界面
root = tk.Tk()
root.title("训练设置")

# 界面布局
tk.Label(root, text="设置训练总轮数:").grid(row=0, column=0, padx=10, pady=10)
epoch_entry = ttk.Entry(root)
epoch_entry.grid(row=0, column=1, padx=10, pady=10)

tk.Label(root, text="每隔多少轮保存模型:").grid(row=1, column=0, padx=10, pady=10)
save_interval_entry = ttk.Entry(root)
save_interval_entry.grid(row=1, column=1, padx=10, pady=10)

train_button = ttk.Button(root, text="开始训练", command=start_training)
train_button.grid(row=2, column=0, columnspan=2, padx=10, pady=20)

root.mainloop()