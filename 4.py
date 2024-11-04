import os
import random
import nltk
from nltk.translate.bleu_score import sentence_bleu
import pandas as pd  # 确保导入 pandas
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from collections import Counter
import re
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from PIL import Image  # 导入 PIL 的 Image 类

# 确保nltk数据已经下载
nltk.download('punkt')


# 构建词典的函数
def build_vocab_from_captions(csv_file, min_freq=1):
    data = pd.read_csv(csv_file)
    captions = data['caption']  # 获取所有报告

    word_counter = Counter()

    # 遍历每个报告，分词并统计词频
    for caption in captions:
        tokens = re.findall(r'\b\w+\b', caption.lower())  # 分词并转换为小写
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
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
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
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_len, embed_size))
        decoder_layer = nn.TransformerDecoderLayer(embed_size, num_heads)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, features, captions, padding_mask=None):
        embeddings = self.embedding(captions) + self.positional_encoding[:, :captions.size(1), :]
        embeddings = embeddings.transpose(0, 1)
        features = features.unsqueeze(1).transpose(0, 1)

        output = self.transformer_decoder(embeddings, features, tgt_key_padding_mask=padding_mask)
        output = output.transpose(0, 1)
        output = self.fc(output)
        return output


# 图像预处理
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0).cuda()  # 添加batch维度并移动到GPU


# 加载模型
def load_model(model_path, word_to_idx):
    checkpoint = torch.load(model_path)
    embed_size = 128
    num_heads = 4
    num_layers = 6
    vocab_size = len(word_to_idx) + 1  # 从词典中获取词汇大小
    max_len = 20

    encoder = CNN_Encoder(embed_size).cuda()
    decoder = Transformer_Decoder(embed_size, num_heads, num_layers, vocab_size, max_len).cuda()

    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])

    encoder.eval()
    decoder.eval()

    return encoder, decoder


# 生成报告
def generate_report(image_path, encoder, decoder, word_to_idx, idx_to_word, max_length=20):
    image_tensor = preprocess_image(image_path)

    with torch.no_grad():
        features = encoder(image_tensor)
        generated_ids = []
        start_token = torch.tensor([[1]]).cuda()  # <START> token

        for _ in range(max_length):
            output = decoder(features, start_token)
            predicted_id = output.argmax(dim=-1).cpu().numpy()[0][-1]  # 获取最后生成的 ID
            generated_ids.append(predicted_id)

            # 如果生成了结束 token，停止生成
            if predicted_id == word_to_idx.get('<END>', None):  # 你可能需要定义 <END> token
                break

            # 更新输入 token
            start_token = torch.tensor([[predicted_id]]).cuda()

        predicted_sentence = ' '.join([idx_to_word.get(id, '<UNK>') for id in generated_ids])
        predicted_sentence = predicted_sentence.replace('<PAD>', '').strip()

    return predicted_sentence


# 主程序
if __name__ == "__main__":
    # 加载词典
    csv_file = 'E:/jsjshij/8222/IUxRay.csv'
    word_to_idx = build_vocab_from_captions(csv_file)
    idx_to_word = {v: k for k, v in word_to_idx.items()}  # 构建 idx_to_word 词典

    # 加载模型
    model_path = 'E:/jsjshij/8222/models/model_epoch_70.pth'  # 请替换为模型的实际路径
    encoder, decoder = load_model(model_path, word_to_idx)  # 传递词典

    # 生成报告
    image_path = 'E:/jsjshij/8222/test_images/CXR1_1_IM-0001-3001.png'  # 替换为要测试的图像路径
    report = generate_report(image_path, encoder, decoder, word_to_idx, idx_to_word)

    print(f"Generated Report: {report}")
