# dos2unix train.sh
CUDA_VISIBLE_DEVICES=6 \
python overall.py \
      --lang c \
      --type uncertain \
      --train_low 1 \
      --train_up 15 \
      --test_low 46 \
      --test_up 60 \
      --k 1000 \
      --loop 8 \