#!/usr/bin/env bash

testfun(){
CUDA_VISIBLE_DEVICES=1 \
python neural_style.py \
--content './image/content/greatwall.jpg' \
--styles './image/style/2-leonid.png' \
--output './image/output/output.png' \
--checkpoint-output ./image/output/mid_images/foo%s.jpg \
--width 1024 \
--style_width 256 \
--style-layer-weight-exp 1 \
--content-weight-blend 1 \
--pooling max \
--content-weight 20 \
--style-weight 500 \
--initial-noiseblend 1 \
--iterations 500 \
--checkpoint-iterations 50 \
--overwrite
}


testfun
