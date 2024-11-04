import os
import glob
import pandas as pd
from xml.dom import minidom
import re
import numpy as np
from tqdm import tqdm

LENGTH = []


def EmptyDrop(data):
    for i in range(len(data)):
        if data.loc[i, 'dir'] == [] or data.loc[i, 'caption'] == []:
            # 如果为空，则删除该行
            data.drop([i], axis=0, inplace=True)
        else:
            data.loc[i, 'dir'] = data.loc[i, 'dir'][0]
            data.loc[i, 'caption'] = data.loc[i, 'caption'][0]
    data.reset_index(drop=True, inplace=True)
    return data


def clean_text(origin_text):
    # 去掉标点和非法字符
    text = re.sub("[^a-zA-Z\s]", " ", origin_text)  # 保留字母和空格
    # 转换为小写
    cleaned_text = text.lower()
    return cleaned_text


def xml2csv(path):
    num = 0
    column_name = ['dir', 'caption']
    xml_csv = pd.DataFrame(columns=column_name)

    # 打印调试信息，检查目录下的文件
    print(f"Reading XML files from: {path}")
    files_in_dir = os.listdir(path)
    print(f"Files in the directory: {files_in_dir}")

    xml_files = glob.glob(path + '/*.xml')
    print(f"Found {len(xml_files)} XML files.")

    if len(xml_files) == 0:
        print("No XML files found. Please check the path.")
        return xml_csv

    # 遍历所有 XML 文件
    for xml_file in tqdm(xml_files):
        # 记录每个 XML 需要保存的信息
        xml_list = []
        # 打开 XML 文档
        dom = minidom.parse(xml_file)
        root = dom.documentElement

        # 获取图像路径
        itemlists = root.getElementsByTagName('parentImage')
        dirAll = []
        for itemlist in itemlists:
            figureId = itemlist.getElementsByTagName('figureId')
            figure = figureId[0].childNodes[0].nodeValue
            ID = itemlist.getAttribute('id')
            figurePath = [figure + ' ' + ID]
            dirAll.extend(figurePath)
        xml_list.append(dirAll)

        # 记录 FINDINGS
        CaptionAll = []
        itemlists = root.getElementsByTagName('AbstractText')
        for i in range(len(itemlists)):
            Label = itemlists[i].getAttribute('Label')
            if Label == 'FINDINGS':
                if len(itemlists[i].childNodes) != 0:
                    text = itemlists[i].childNodes[0].nodeValue
                    text = clean_text(text)
                    text = text.replace('.', '')
                    text = text.replace(',', '')
                    text = [text + '']
                    CaptionAll.extend(text)

        if len(CaptionAll) >= 1:
            LENGTH.append(len(CaptionAll[0].split(' ')))
        xml_list.append(CaptionAll)
        xml_csv.loc[num] = [item for item in xml_list]
        num += 1

    return xml_csv


def main():
    # 修改路径为你的 reports 文件夹，使用 os.path.join 来构建路径
    xml_path = os.path.join('E:\\', 'jsjshij', '8222', 'reports')

    # 生成 CSV
    csv = xml2csv(xml_path)

    # 检查 CSV 是否为空
    if csv.empty:
        print("No data processed, CSV is empty.")
        return

    csv_cleaned = EmptyDrop(csv)

    # 确保目标路径存在
    output_dir = os.path.join('E:\\', 'jsjshij', '8222')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # 创建不存在的目录

    # 保存 CSV 文件
    csv_cleaned.to_csv(os.path.join(output_dir, 'IUxRay.csv'), index=None)
    print("CSV file saved successfully.")


if __name__ == '__main__':
    main()
