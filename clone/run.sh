# dos2unix train.sh
CUDA_VISIBLE_DEVICES=5 \
python pipeline.py \
      --lang c \
      --train_low 1 \
      --train_up 15 \
      --test_low 46 \
      --test_up 60 \

CUDA_VISIBLE_DEVICES=5 \
python train.py \
      --lang c \