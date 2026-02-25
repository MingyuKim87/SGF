#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import shutil
import argparse
import shlex
from pathlib import PurePosixPath

class ArgsWithComments(argparse.ArgumentParser):
    """
    '@file' 로 인자 로딩 + '# ...' 주석/공백 허용
    """
    def __init__(self, *a, **kw):
        kw.setdefault("fromfile_prefix_chars", "@")
        kw.setdefault("formatter_class", argparse.ArgumentDefaultsHelpFormatter)
        super().__init__(*a, **kw)

    def convert_arg_line_to_args(self, arg_line):
        s = arg_line.strip()
        if not s or s.startswith("#"):
            return []
        # 공백/따옴표를 쉘과 동일하게 처리
        return shlex.split(s)

def to_relative(path, abs_prefix=None, anchor=None, prepend=""):
    """
    path(문자열)를 상대경로로 변환.
    1) abs_prefix 로 시작하면 그 부분 제거
    2) 아니면 anchor('train/images_large') 이후만 남김
    3) 둘 다 실패하면 파일명만 남김
    마지막에 prepend가 있으면 앞에 붙임
    """
    s = str(PurePosixPath(path))  # '/' 구분자 강제

    if abs_prefix:
        abs_prefix_norm = str(PurePosixPath(abs_prefix)).rstrip('/') + '/'
        if s.startswith(abs_prefix_norm):
            rel = s[len(abs_prefix_norm):].lstrip('/')
            return str(PurePosixPath(prepend) / rel) if prepend else rel

    if anchor:
        anchor_norm = '/' + anchor.strip('/') + '/'
        i = s.find(anchor_norm)
        if i != -1:
            rel = s[i+1:]  # 맨 앞 '/' 제거
            return str(PurePosixPath(prepend) / rel) if prepend else rel

    rel = PurePosixPath(s).name
    return str(PurePosixPath(prepend) / rel) if prepend else rel

def rewrite_json(in_path, out_path=None,
                 abs_prefix="/fs/cml-projects/diffusion_rep/data/laion_10k_random",
                 anchor="train/images_large",
                 prepend="datasets/LAION_10k"):
    """
    JSON의 {이미지경로: [캡션들]} 구조에서 key를 상대경로로 바꿔 저장.
    """
    in_path = str(in_path)

    if out_path is None:
        backup = in_path + ".bak"
        shutil.copyfile(in_path, backup)
        print(f"[백업] {backup}")

    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError("JSON 최상위는 dict(객체)여야 합니다.")

    new_data = {}
    collisions = []
    for k, v in data.items():
        new_key = to_relative(k, abs_prefix=abs_prefix, anchor=anchor, prepend=prepend)
        if new_key in new_data:
            collisions.append((new_key, k))
        else:
            new_data[new_key] = v

    if collisions:
        print(f"[경고] 키 충돌 {len(collisions)}건 (첫 항목만 유지). 예시 5개:")
        for nk, orig in collisions[:5]:
            print(" -", nk, "<-", orig)

    out_path = out_path or in_path
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2, sort_keys=True)
    print(f"[완료] {in_path} -> {out_path}")

def main():
    p = ArgsWithComments(description="JSON 이미지 경로를 절대→상대 경로로 변환")
    p.add_argument("--in", dest="in_path", required=True, help="입력 JSON 경로")
    p.add_argument("--out", dest="out_path", default=None, help="출력 JSON 경로(미지정 시 in-place)")
    p.add_argument("--abs-prefix", dest="abs_prefix",
                   default="/fs/cml-projects/diffusion_rep/data/laion_10k_random",
                   help="기존 절대경로 공통 접두사")
    p.add_argument("--anchor", dest="anchor", default="train/images_large",
                   help="경로 내에서 유지할 하위 경로 기준(예: train/images_large)")
    p.add_argument("--prepend", dest="prepend", default="datasets/LAION_10k",
                   help="새 상대경로 앞에 붙일 접두사(예: datasets/LAION_10k)")
    args = p.parse_args()

    rewrite_json(args.in_path, args.out_path,
                 abs_prefix=args.abs_prefix, anchor=args.anchor, prepend=args.prepend)

if __name__ == "__main__":
    main()
