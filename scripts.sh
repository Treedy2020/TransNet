python ./main.py \
  --data-dir './COST2100' \ # root dir path for cost2100
  --scenario 'in' \ # in or out
  --epochs 1000 \
  --d_model 64 \  # dimension of feature in transformer
  --batch-size 200 \
  --workers 3 \
  --cr 4 \
  --scheduler const \ const or cosine
  --gpu 0 \
  2>&1 | tee log.out