#!/usr/bin/env bash

testfun(){
CUDA_VISIBLE_DEVICES=1 \
python neural_style.py \
--content ${content_path}${1} \
--styles ${style_path}${2} \
--output "./image/output/${1}_${2}_.png" \
--checkpoint-output ./image/mid_image/foo%s.jpg \
--width 1024 \
--square_shape \
--style_width 512 \
--iterations 500 \
--checkpoint-iterations 5000 \
--content-weight 5e0 \
--style-weight 5e2 \
--overwrite
}

content_path='./image/content/'
style_path='./image/style/'
for content in `ls ${content_path}`
do
    for style in `ls ${style_path}`
    do
        echo ${content} ${style}
        testfun ${content} ${style}
    done
done


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
