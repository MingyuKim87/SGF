import os
import re
import pandas as pd
from glob import glob
from PIL import Image
from collections import defaultdict
from typing import Callable, Optional
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader


__DATASET__ = {}
# allow for webp
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

def pil_loader_rgb(path: str) -> Image.Image:
    # PIL for webp
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")

def load_imagenet_classnames(root: str) -> pd.DataFrame:
    """
    ImageNet classnames.txt 파일을 읽어서 DataFrame으로 반환합니다.
    
    Parameters
    ----------
    txt_path : str
        classnames.txt 파일 경로
    
    Returns
    -------
    pd.DataFrame
        wnid, name, clean_name 컬럼을 가진 DataFrame
    """
    rows = []
    
    txt_path = os.path.join(root, "classnames.txt")

    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            wnid, name = line.split(" ", 1)  # wnid, 나머지 이름
            rows.append((wnid, name))

    df = pd.DataFrame(rows, columns=["wnid", "name"])

    # 이름 정리
    def clean_name(s: str) -> str:
        s = s.strip()
        s = s.split(",")[0]                # 콤마 있으면 첫 번째만
        s = re.sub(r"\s*\(.*?\)\s*", "", s) # 괄호 제거
        return s.lower()

    df["clean_name"] = df["name"].apply(clean_name)

    return df

# for memorization
def is_valid_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in IMG_EXTS

def register_dataset(name: str):
    def wrapper(cls):
        if __DATASET__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __DATASET__[name] = cls
        return cls
    return wrapper

def get_dataset(name: str, root: str, **kwargs):
    if __DATASET__.get(name, None) is None:
        raise NameError(f"Dataset {name} is not defined.")
    
    # exceptio treatment: transform -> transforms
    if "transform" not in kwargs and "transforms" in kwargs:
        kwargs["transform"] = kwargs.pop("transforms")
    
    new_dataset = __DATASET__[name](root=root, **kwargs)
    return new_dataset

def get_all_imgs(dataloader):
    all_imgs = []
    for images in dataloader:
        all_imgs.append(images)
    all_images = torch.cat(all_imgs, dim=0)
    return all_images

def get_all_imgs_imageFolder(dataloader):
    '''for imageFolder'''
    
    all_imgs = []
    for images, _ in dataloader:
        all_imgs.append(images)
    all_images = torch.cat(all_imgs, dim=0)
    return all_images

def get_class_subset_loader(dataset, target_class, batch_size=64, num_workers=4, shuffle=False):
    """
    ImageFolder 기반 dataset에서 특정 클래스(wnid 또는 class name)의 DataLoader를 반환합니다.
    """
    # target_class는 wnid (예: "n01534433") 라고 가정
    if target_class not in dataset.class_to_idx:
        raise ValueError(f"{target_class} not found in dataset.classes")

    cls_idx = dataset.class_to_idx[target_class]

    # dataset.samples: [(path, label), ...]
    indices = [i for i, (_, y) in enumerate(dataset.samples) if y == cls_idx]

    subset = Subset(dataset, indices)

    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

def get_all_imgs_by_class(dataloader):
    """
    dataloader: ImageFolder 기반 DataLoader
    return: dict { class_name: torch.Tensor (Nc, C, H, W) }
    """
    # initialize a list for each class_name 
    imgs_dict = defaultdict(list)
    idx_to_class = {v: k for k, v in dataloader.dataset.class_to_idx.items()}

    for images, labels in dataloader:  # images: (B, C, H, W), labels: (B,)
        for i in range(len(labels)):
            class_name = idx_to_class[int(labels[i])]
            imgs_dict[class_name].append(images[i].unsqueeze(0))  # (1, C, H, W)

    # revert list → Tensor
    for class_name in imgs_dict:
        imgs_dict[class_name] = torch.cat(imgs_dict[class_name], dim=0)

    return imgs_dict

def get_all_imgs_by_target_class(dataloader, target_class=None):
    '''Deprecated'''
    idx = dataloader.dataset.class_to_idx[str(target_class)]
    all_imgs = []
    for images, labels in dataloader:  # (B, C, H, W), (B,)
        mask = labels == idx           
        if mask.any():
            all_imgs.append(images[mask])  
    if all_imgs:
        all_images = torch.cat(all_imgs, dim=0)
        return all_images
    else:
        return torch.empty(0)
    
def get_dataloader(dataset: VisionDataset,
                   batch_size: int, 
                   num_workers: int, 
                   train: bool):
    dataloader = DataLoader(dataset, 
                            batch_size, 
                            shuffle=train, 
                            num_workers=num_workers, 
                            drop_last=train)
    return dataloader

def get_transform(name: str, **kwargs):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    return transform

@register_dataset(name='memorization')
class ImgDataset(ImageFolder):
    def __init__(self, root:str, class_info:str, transform=None):
        super().__init__(
            root=os.path.join(root, class_info),
            transform=transform,
            loader=pil_loader_rgb,
            is_valid_file=is_valid_file,
        )

@register_dataset(name="diversity")
class ImagenetDataset(ImageFolder):
    def __init__(self, root:str, class_info:str, transform=None):
        super().__init__(
            root=os.path.join(root, class_info),
            transform=transform,
        )

@register_dataset(name='nudity')
@register_dataset(name='inappropriate')
class NudityDataset(VisionDataset):
    def __init__(self, root: str, class_info:str, transforms: Optional[Callable]=None):
        super().__init__(root, transforms)
        root_path = os.path.join(root, class_info)
        # self.fpaths = sorted(glob(f'{root_path}/*.png', recursive=True))
        self.fpaths = sorted(glob(f'{root_path}/*.png', recursive=True) +
                     glob(f'{root_path}/*.jpg', recursive=True))
        
        # VRAM out of memory
        if len(self.fpaths) > 3200:
            self.fpaths = self.fpaths[:3200]
        
        assert len(self.fpaths) > 0, "File list is empty. Check the root."
        
    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, index: int):
        fpath = self.fpaths[index]
        img = Image.open(fpath).convert('RGB')
        
        if self.transforms is not None:
            img = self.transforms(img)
        return img

# check

def check_empty_classes(root, valid_exts={".jpg", ".jpeg", ".png", ".webp"}):
    """
    root: ImageFolder의 root 디렉토리
    valid_exts: 허용하는 확장자
    """
    empty_classes = []
    all_classes = []
    for cls_name in sorted(os.listdir(root)):
        cls_path = os.path.join(root, cls_name)
        if not os.path.isdir(cls_path):
            continue
        all_classes.append(cls_name)

        # cls_path 아래에 이미지가 있는지 검사
        has_image = False
        for fname in os.listdir(cls_path):
            if os.path.splitext(fname)[1].lower() in valid_exts:
                has_image = True
                break
        if not has_image:
            empty_classes.append(cls_name)

    return all_classes, empty_classes

import os
from PIL import Image

def clean_broken_images(root, valid_exts={".jpg", ".jpeg", ".png", ".webp"}):
    """
    root: 최상위 데이터셋 폴더 (예: memorization_sd14_candidates)
    valid_exts: 허용 확장자 집합
    """
    removed_files = []

    for subdir, _, files in os.walk(root):
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in valid_exts:
                continue

            fpath = os.path.join(subdir, fname)
            try:
                # 이미지 열어보고 바로 닫기
                with Image.open(fpath) as img:
                    img.verify()  # 파일 무결성 체크
            except Exception as e:
                print(f"⚠️ 삭제: {fpath} (에러: {e})")
                os.remove(fpath)
                removed_files.append(fpath)

    print(f"\n총 {len(removed_files)}개 파일 삭제 완료")
    return removed_files



if __name__ == "__main__":
    root = "./datasets/imagenet/images"
    class_info = "train"
    tranform_fn = get_transform("default")

    train_dataset = ImagenetDataset(root=root, class_info=class_info, transform=tranform_fn)

    print(train_dataset.classes[:10])  # 일부 클래스 이름
    print(train_dataset.class_to_idx)  # 클래스 -> 인덱스 매핑

    