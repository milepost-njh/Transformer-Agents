docker run --gpus all -idt \
  --name trs \
  --network host \
  -v /data2/workspace/yszhang/train_transformers:/workspace \
  --shm-size=16g \
  trs-migrated:latest