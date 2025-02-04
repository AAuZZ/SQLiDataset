import json

def convert_jsonl_to_json(jsonl_file_path, output_json_file_path):
    data_list = []
    
    # 读取jsonl文件并解析每一行为一个json对象
    with open(jsonl_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip():  # 确保不是空行
                data_list.append(json.loads(line))
    
    # 将列表写入到新的json文件中
    with open(output_json_file_path, 'w', encoding='utf-8') as outfile:
        json.dump(data_list, outfile, ensure_ascii=False, indent=4)

# 使用示例
convert_jsonl_to_json('./sqli1.jsonl', 'sqli1.json')