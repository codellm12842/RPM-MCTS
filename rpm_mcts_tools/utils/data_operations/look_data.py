import json


dataset_path = '/path/to/input.jsonl'
output_path = dataset_path.replace('.json', '.txt')

with open(dataset_path, 'r') as f:
    if dataset_path.endswith('.jsonl'):
        data = [json.loads(line) for line in f]
    else:
        data = json.load(f)
    print(f"all process {len(data)} datas")


excluded_keys = ['source_file', 'test_imports', 'problem', 'prompt']
with open(output_path, 'w') as f:
    for i, item in enumerate(data):
        for key, value in item.items():
            if key not in excluded_keys:
                f.write(f"{key}: {value}\n\n")
        f.write('----------------------------------------\n\n')

print(f"Data has been saved to {output_path}")
