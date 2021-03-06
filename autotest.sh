#!/usr/bin/env bash

testfun(){
CUDA_VISIBLE_DEVICES=1 \
python neural_style.py \
--content ${content_path}${1} \
--styles ${style_path}${2} \
--output ${output_path}"${1}_${2}_.png" \
--checkpoint-output ./image/mid_image/foo%s.jpg \
--width 1024 \
--square_shape \
--style_width 512 \
--iterations 500 \
--checkpoint-iterations 5000 \
--content-weight 5e0 \
--content-weight-blend 0.4 \
--style-weight 1e1 \
--overwrite
}
#`ls ${style_path}`
# 默认参数
#CONTENT = './image/content/greatwall.jpg'
#STYLE = ['./image/style/2-leonid.png']
#OUTPUT = './image/output/output_test.png'
#CHECKPOINT_OUTPUT = './image/mid_image/foo%s.jpg'
#CONTENT_WEIGHT = 5e0
#CONTENT_WEIGHT_BLEND = 1
#STYLE_WEIGHT = 5e2
#TV_WEIGHT = 1e2
#STYLE_LAYER_WEIGHT_EXP = 1
#LEARNING_RATE = 1e1
#BETA1 = 0.9
#BETA2 = 0.999
#EPSILON = 1e-08
#STYLE_SCALE = 1.0
#ITERATIONS = 1000

content_path='./image/special_content/'
style_path='./image/style/'
output_path='./image/output_archive/leonid/'
content_list=`ls ${content_path}`
style_list=(2-leonid.png)
for content in ${content_list}
do
    for style in ${style_list}
    do
        echo ${content_list[*]}
        echo ${style_list[*]}
        echo ${content} ${style}
        testfun ${content} ${style}
    done
done

python img_grid.py \
--contents_path ${content_path} \
--styles_path ${style_path} \
--output_path ${output_path} \
--content_list ${content_list[*]} \
--style_list ${style_list[*]}
