#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json, re
from pathlib import Path

def load_classnames(path):
    """
    classnames.txt 예:
      n01440764 tench
      n01608432 kite (bird of prey)
    반환: {wnid: "tench", ...}
    """
    m = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        mo = re.match(r"^(n\d{8})\s+(.+)$", s)
        if not mo:
            continue
        wnid, name_str = mo.group(1), mo.group(2).strip()
        name = name_str.split(",")[0].strip()  # 콤마 이후 별칭 제거
        m[wnid] = name
    return m

def make_prompt(name):
    # 따옴표 없이 출력
    return f"An image of {name}"

def make_prompt_nolevel(name):
    # 따옴표 없이 출력
    return f"An image of"


def build_json(dataset_root, classmap, splits=("train","val"),
               prepend="", exts=(".jpg",".jpeg",".png",".JPG",".JPEG",".PNG")):
    root = Path(dataset_root)
    out = {}
    for split in splits:
        d = root / split
        if not d.is_dir():
            print(f"[경고] '{split}' 디렉터리 없음: {d}")
            continue
        for wnid_dir in sorted([p for p in d.iterdir() if p.is_dir()]):
            wnid = wnid_dir.name
            cls_name = classmap.get(wnid)
            if not cls_name:
                print(f"[주의] classnames에 없음 -> skip: {wnid}")
                continue
            for img in sorted(wnid_dir.rglob("*")):
                if img.is_file() and img.suffix in exts:
                    rel = img.relative_to(root)
                    key = str((Path(prepend) / rel) if prepend else rel)
                    # out[key] = [make_prompt(cls_name)]
                    out[key] = [make_prompt_nolevel(cls_name)]
    return out

def main():
    ap = argparse.ArgumentParser(description="Imagenette → {path: [prompt]} JSON 생성")
    ap.add_argument("--root", required=True, help="데이터셋 루트 (예: imagenette2-320)")
    ap.add_argument("--classnames", required=True, help="classnames.txt 경로")
    ap.add_argument("--splits", default="train,val", help="예: train 또는 train,val")
    ap.add_argument("--prepend", default="", help="키 앞에 붙일 접두사 (예: datasets/imagenette2-320)")
    ap.add_argument("--out", required=True, help="출력 JSON")
    args = ap.parse_args()

    classmap = load_classnames(args.classnames)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    data = build_json(args.root, classmap, splits=tuple(splits), prepend=args.prepend)

    Path(args.out).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[완료] {len(data):,}개 이미지 캡션 저장 → {args.out}")

if __name__ == "__main__":
    main()
