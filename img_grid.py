import os
from PIL import Image
from argparse import ArgumentParser

print('\033[1;31;40m')
print('Run img_grid.py')
print('\033[0m')

parser = ArgumentParser()
parser.add_argument('--contents_path')
parser.add_argument('--styles_path')
parser.add_argument('--output_path')
parser.add_argument('--content_list', nargs='+')
parser.add_argument('--style_list', nargs='+')
options = parser.parse_args()
print(options.content_list)
print(options.style_list)
contents = options.content_list
styles = options.style_list
contents_path = options.contents_path
styles_path = options.styles_path
output_path = options.output_path

img_name = contents[0] + '_' + styles[0] + '_.png'
img = Image.open(os.path.join(output_path, img_name))
width = img.size[0]
height = width
width_sum = width * (len(styles) + 1)
height_sum = height * (len(contents) + 1)
img_sum = Image.new('RGB', (width_sum, height_sum), (255, 255, 255))

for i, content in enumerate(contents):
    img = Image.open(os.path.join(contents_path, content))
    img = img.resize((width, height))
    img_sum.paste(img, (0, i * height + height))

for i, style in enumerate(styles):
    img = Image.open(os.path.join(styles_path, style))
    img = img.resize((width, height))
    img_sum.paste(img, (i * width + width, 0))

for i, content in enumerate(contents):
    for j, style in enumerate(styles):
        img_name = content + '_' + style + '_.png'
        if os.path.exists(os.path.join(output_path, img_name)):
            img = Image.open(os.path.join(output_path, img_name))
            img = img.resize((width, height))
            img_sum.paste(img, (j * width + width, i * height + height))
        else:
            print(img_name)

img_sum.save(os.path.join(output_path, '0img_sum.png'))
print('Save to %s' %os.path.join(output_path, '0img_sum.png'))
