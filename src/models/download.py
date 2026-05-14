from huggingface_hub import hf_hub_download
import os

os.makedirs('models', exist_ok=True)

filepath = hf_hub_download(
    repo_id='juliozhao/DocLayout-YOLO-DocLayNet-Docsynth300K_pretrained',
    filename='doclayout_yolo_doclaynet_imgsz1120_docsynth_pretrain.pt',
    local_dir='models',
    local_dir_use_symlinks=False
)

print(f'Model is saved: {filepath}')