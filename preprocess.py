import pandas as pd
import re
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
import torch

# 读取CSV文件
read_csv=pd.read_csv("balanced_ai_human_prompts.csv")# 文件名而不是路径
print(read_csv.head())

# 分别读取CSV文件中的两列：text和generated（人类/AI创作标签）
texts=read_csv['text'].tolist()
labels=read_csv['generated'].tolist()
print(texts[:5])
print(labels[:5])   

# 数据清洗，去除缺失项，去除重复项，去除异常值
data=pd.DataFrame({'text':texts,'generated':labels})
data.dropna(inplace=True)
data.drop_duplicates(inplace=True)  
texts=data['text'].tolist()
labels=data['generated'].tolist()
print(f"数据清洗后，剩余样本数：{len(texts)}")

# 文本清洗，去除多余空格、标点符号，大写转换成小写
def clean_text(text):
    text = text.lower()  # 转换为小写
    text = re.sub(r'\s+', ' ', text)  # 去除多余空格
    text = re.sub(r'[^\w\s]', '', text)  # 去除标点符号
    return text.strip()
texts = [clean_text(t) for t in texts]
print(f"文本清洗后，前5条样本：{texts[:5]}")

# 将所有的text各自tokenize，并记录长度等基本信息
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenized_texts = [tokenizer.tokenize(t) for t in texts]
text_lengths = [len(t) for t in tokenized_texts]
print(f"前5条样本的tokenized结果：{tokenized_texts[:5]}")
print(f"前5条样本的长度：{text_lengths[:5]}")
print(f"文本长度的基本统计信息：平均长度={sum(text_lengths)/len(text_lengths):.2f}, 最大长度={max(text_lengths)}, 最小长度={min(text_lengths)}")

# 通过encode_plus添加 [CLS], [SEP], padding, attention mask
MAX_LEN = 128

input_ids = []
attention_masks = []

for txt in data['text']:
    encoding = tokenizer.encode_plus(
        txt,
        add_special_tokens=True,      # 加 [CLS] 和 [SEP]
        max_length=MAX_LEN,
        padding='max_length',         # 补到最大长度
        truncation=True,              # 截断超长文本
        return_attention_mask=True,   # 生成 attention mask
        return_tensors='pt'           # 返回 PyTorch tensor
    )
    input_ids.append(encoding['input_ids'])
    attention_masks.append(encoding['attention_mask'])

# 转成 tensor（方便后面模型训练）
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels_t = torch.tensor(data['generated'].values)

print(f"input_ids shape: {input_ids.shape}")
print(f"attention_masks shape: {attention_masks.shape}")
print(f"labels shape: {labels_t.shape}")

# 划分训练/验证/测试集
train_idx, val_idx = train_test_split(
    range(len(data)),
    test_size=0.2,
    random_state=42,
    stratify=data['generated']
)

train_inputs = input_ids[train_idx]
train_masks = attention_masks[train_idx]
train_labels = labels_t[train_idx]

val_inputs = input_ids[val_idx]
val_masks = attention_masks[val_idx]
val_labels = labels_t[val_idx]

print(f"训练样本数: {len(train_labels)}, 验证样本数: {len(val_labels)}")

# 至此数据预处理完成

# 保存预处理结果（文本+标签）
data.to_csv("processed_balanced_ai_human.csv", index=False)

# 保存 tensor 数据
torch.save((train_inputs, train_masks, train_labels), "train_data.pt")
torch.save((val_inputs, val_masks, val_labels), "val_data.pt")