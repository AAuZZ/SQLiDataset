import json

def validate_jsonl_format(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        for line_number, line in enumerate(file, start=1):
            try:
                data = json.loads(line)
                
                if not isinstance(data.get('conversations'), list):
                    print(f"Line {line_number} is invalid: 'conversations' should be a list.")
                    continue
                
                for conv in data['conversations']:
                    if not (isinstance(conv, dict) and 'from' in conv and 'value' in conv):
                        print(f"Line {line_number} is invalid: each conversation must have 'from' and 'value' attributes.")
                        break
            except json.JSONDecodeError:
                print(f"Line {line_number} is invalid JSON.")

# Example usage:
validate_jsonl_format('./sqli3.jsonl')



