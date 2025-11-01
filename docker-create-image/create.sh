docker run --gpus all -idt \
  --name trs \
  --network host \
  -v /data4/yszhang/gits/Transformer-Agents:/workspace \
  --shm-size=16g \
  nvcr.io/nvidia/pytorch:25.10-py3