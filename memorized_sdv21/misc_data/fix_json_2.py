import json

input_path = "datasets/imagenette2-320/imagenette_prompts_nolevel.json"
output_path = "datasets/imagenette2-320/imagenette_prompts_nolevel_rev.json"

# JSON 읽기
with open(input_path, "r") as f:
    data = json.load(f)

# key 이름 교체
fixed_data = {}
for key, value in data.items():
    new_key = key.replace("imagenettte2-320", "imagenette2-320")
    fixed_data[new_key] = value

# JSON 저장
with open(output_path, "w") as f:
    json.dump(fixed_data, f, indent=2)

print(f"수정 완료! {output_path} 에 저장됨")