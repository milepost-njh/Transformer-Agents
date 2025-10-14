# docker-create-image

在 Docker 中基于 Base Image 创建自定义 Docker-Image

## 项目目录结构
* README.md -- 项目指引

* create_custom_container.sh -- create Pytorch Docker container script



## Pytorch  - NGC 2505 镜像构建
* 1、拉取代码
```shell
git clone https://github.com/milepost-njh/Transformer-Agents.git
```

* 2、拉取 Base Docker Image
```shell
# 拉取 Nvidia-NGC-Pytorch Version：25.05
# eg: 
docker pull nvcr.io/nvidia/pytorch:25.05-py3
```

* 3、在 create_custom_container.sh 文件
```shell
# 修改 create_custom_container.sh ：
cd docker-create-image
vim create_custom_container.sh
# 注意：在最后一行修改 Base-Image tag:version
```

* 4、创建 Docker Custom-Container
```shell 
# eg： 
# bash create_custom_container.sh pytorch_ngc_2505  nvcr.io/nvidia/pytorch  25.05-py3  20004 20005 20006
```

* 5、进入 Custom-Container
```shell
# eg:
docker exec -it pytorch_ngc_2505 bash
```