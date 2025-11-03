# Docker容器迁移指南

## 概述

本指南用于将远程服务器上的Docker容器迁移到当前服务器，并映射到本地项目目录。

## 手动迁移步骤

按照以下步骤手动完成迁移：

### 步骤1: 在远程服务器上提交容器为镜像

```bash
# 在远程服务器上执行（你有直接访问权限）
docker commit trs trs-migrated:latest
```

### 步骤2: 在远程服务器上导出镜像

```bash
# 仍在远程服务器上执行
docker save trs-migrated:latest -o ./trs-container.tar
```

### 步骤3: 在远程服务器上传镜像到OSS

```bash
# 仍在远程服务器上执行
# 上传到OSS（需要先安装ossutil工具）
ossutil cp ./trs-container.tar oss://mr-projects/ys/transformers/trs-container.tar

# 如果ossutil未安装，可以先下载安装：
# wget https://gosspublic.alicdn.com/ossutil/1.7.14/ossutil64
# chmod 755 ossutil64
# ./ossutil64 config
```

### 步骤4: 在远程服务器上清理镜像和临时文件

```bash
# 仍在远程服务器上执行
# 删除导出的tar文件
rm -f ./trs-container.tar

# 删除创建的镜像
docker rmi trs-migrated:latest
```

### 步骤5: 在本地从OSS下载镜像

```bash
# 在本地服务器上执行
# 从OSS下载镜像文件（需要先安装ossutil工具）
ossutil cp oss://mr-projects/ys/transformers/trs-container.tar ./trs-container.tar

# 如果ossutil未安装，可以先下载安装：
# wget https://gosspublic.alicdn.com/ossutil/1.7.14/ossutil64
# chmod 755 ossutil64
# ./ossutil64 config
```

### 步骤6: 在本地加载镜像

```bash
# 在本地服务器上执行
docker load -i ./trs-container.tar
```

### 步骤7: 创建新容器并映射项目目录

```bash
# 在本地服务器上执行

# 创建新容器
docker run --gpus all -idt \
  --name trs \
  --network host \
  -v /data2/workspace/yszhang/train_transformers:/workspace \
  --shm-size=16g \
  trs-migrated:latest
```

### 步骤8: 清理本地临时文件

```bash
# 在本地服务器上执行
# 删除下载的tar文件
rm -f /tmp/trs-container.tar

# 可选：删除OSS上的镜像文件（如果需要节省存储空间）
# ossutil rm oss://mr-projects/ys/transformers/trs-container.tar
```

## 验证迁移

### 检查镜像

```bash
docker images | grep trs-migrated
```

### 检查容器

```bash
docker ps | grep trs
```

### 进入容器验证

```bash
docker exec -it trs /bin/bash
cd /workspace
ls -la
# 应该能看到项目的所有文件
```

## 常见问题

### 1. ossutil工具安装和配置

如果未安装ossutil，需要先安装和配置：

```bash
# 下载ossutil（Linux 64位）
wget https://gosspublic.alicdn.com/ossutil/1.7.14/ossutil64
chmod 755 ossutil64

# 配置ossutil（需要AccessKey ID和AccessKey Secret）
./ossutil64 config

# 或者将ossutil64移动到系统PATH中
sudo mv ossutil64 /usr/local/bin/ossutil64
```

配置ossutil时需要提供：
- Endpoint: OSS服务的访问域名
- AccessKey ID: 阿里云访问密钥ID
- AccessKey Secret: 阿里云访问密钥Secret

### 2. 磁盘空间不足

导出镜像可能很大（>10GB），确保有足够的磁盘空间：

```bash
df -h /tmp
```

### 3. OSS上传/下载速度

OSS上传和下载速度取决于网络带宽。如果文件很大（>10GB），可能需要较长时间。可以使用以下命令查看上传进度：

```bash
# 查看上传进度（使用ossutil的进度显示）
ossutil cp /tmp/trs-container.tar oss://mr-projects/ys/transformers/trs-container.tar --update
```

### 4. 容器已存在

如果本地已有同名容器，需要先删除：

```bash
docker stop trs
docker rm trs
```

### 5. 权限问题

确保有Docker执行权限：

```bash
sudo usermod -aG docker $USER
# 然后重新登录
```

## 清理临时文件

迁移完成后，清理临时文件：

```bash
# 本地服务器（删除下载的tar文件）
rm -f /tmp/trs-container.tar

# 远程服务器（步骤4已包含，这里仅作参考）
# rm -f /tmp/trs-container.tar
# docker rmi trs-migrated:latest

# OSS上的文件（可选，如果不再需要可以删除）
# ossutil rm oss://mr-projects/ys/transformers/trs-container.tar
```

**注意**: 
- 步骤4已包含远程服务器的清理操作（删除tar文件和镜像）
- 步骤8已包含本地tar文件的清理
- OSS上的文件可根据需要决定是否删除

## 容器管理命令

### 启动容器
```bash
docker start trs
```

### 停止容器
```bash
docker stop trs
```

### 重启容器
```bash
docker restart trs
```

### 进入容器
```bash
docker exec -it trs /bin/bash
```

### 查看容器日志
```bash
docker logs trs
```

### 删除容器
```bash
docker stop trs
docker rm trs
```

### 删除镜像
```bash
docker rmi trs-migrated:latest
```

