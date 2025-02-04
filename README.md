# 概述
本数据集用于训练大模型进行SQL注入的监督微调。数据集包含了一系列模拟的SQL注入场景，每个场景包括用户输入和模型响应的对话记录。通过这些对话记录，模型可以学习如何识别和利用SQL注入漏洞。

# 数据集结构
数据集以JSON格式存储，每个JSON对象代表一个独立的SQL注入场景，包含多个对话回合（conversations）。每个对话回合包含两个字段：
- `from`: 表示对话的发起方，可以是`human`（用户）或`gpt`（模型）。
- `value`: 表示对话的具体内容。

加载数据集
你可以使用Python加载数据集。以下是一个示例代码：

python
import json

# 读取JSONL文件
with open('sqli1.jsonl', 'r', encoding='utf-8') as file:
    data = [json.loads(line) for line in file]

# 打印前两个场景
for i, scenario in enumerate(data[:2]):
    print(f"Scenario {i+1}:")
    for conversation in scenario['conversations']:
        print(f"  {conversation['from']}: {conversation['value']}")
数据预处理
在训练模型之前，可能需要对数据进行预处理，例如去除特殊字符、分词等。以下是一个简单的预处理示例：

python
import re

def preprocess_text(text):
    # 去除特殊字符
    text = re.sub(r'[^\w\s]', '', text)
    # 转换为小写
    text = text.lower()
    return text

# 预处理数据
preprocessed_data = []
for scenario in data:
    preprocessed_conversations = []
    for conversation in scenario['conversations']:
        preprocessed_conversations.append({
            'from': conversation['from'],
            'value': preprocess_text(conversation['value'])
        })
    preprocessed_data.append({'conversations': preprocessed_conversations})

# 打印预处理后的前两个场景
for i, scenario in enumerate(preprocessed_data[:2]):
    print(f"Preprocessed Scenario {i+1}:")
    for conversation in scenario['conversations']:
        print(f"  {conversation['from']}: {conversation['value']}")
模型训练
以下是一个简单的模型训练示例，使用预处理后的数据集：

python
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments

# 加载预训练的GPT-2模型和分词器
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# 将对话转换为模型输入格式
def encode_conversations(scenarios):
    inputs = []
    for scenario in scenarios:
        conversation = ' '.join([f"{conv['from']}: {conv['value']}" for conv in scenario['conversations']])
        inputs.append(conversation)
    return tokenizer(inputs, return_tensors='pt', padding=True, truncation=True)

# 编码数据
encoded_data = encode_conversations(preprocessed_data)

# 定义训练参数
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=10_000,
    save_total_limit=2,
)

# 定义Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_data['input_ids'],
)

# 开始训练
trainer.train()
数据集贡献
如果你有更多高质量的SQL注入场景，欢迎贡献到这个数据集。请按照以下格式提交你的数据：

json
{
    "conversations": [
        {
            "from": "human",
            "value": "你的对话内容"
        },
        {
            "from": "gpt",
            "value": "模型的响应内容"
        }
    ]
}
许可证
本数据集遵循MIT许可证。请在使用和分发时遵守相关条款。