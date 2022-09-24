#!/bin/bash

python test.py \
--project yolov3_rtts_10c_deyolo \
--name deyolo \
--model deyolo.DEYOLO \
--img_size_test 544 544 \
--batch_size 8 \
--data ./data/rtts_5c.yaml \
--hyp ./hyp/hyp.voc.scratch.yaml \
--verbose \
--nms_thres 0.5 \
--conf_thres 0.001 \
--checkpoint pretrained_models/deyolo_foggy/best.pt
