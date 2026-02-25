from PIL import Image


def concat_h(*argv, pad=0):
    width = 0
    height = 0
    count = len(argv)

    for img in argv:
        width += img.width
        height = max(height, img.height)

    dst = Image.new('RGB', (width + (count-1)*pad, height))
    start = 0
    for i, img in enumerate(argv):
        dst.paste(img, (start, 0))
        start += img.width + pad
    return dst

