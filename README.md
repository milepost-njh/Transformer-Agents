# Transformer-Agents

✈️  构建Transformer框架代码并结合SOTA模型结构； LLM Agent相关工作

- 基于 Transformer 多语言模型分类问题的全流程项目。

## 项目目录结构
* readme.md -- 项目指引

* core -- 核心模块：检查点、数据、损失函数、模型、优化器

* create-dataset -- 构造数据脚本文件夹

* doc -- 历史文件

* docker-create-image -- 项目容器化构建

* script -- 启动脚本

* test -- 测试脚本

* train -- 模型训练



## 镜像构建
```shell
# Pytorch 镜像版本对应可参考：
https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html
```
* 1、Transformer 模型容器环境搭建
```shell
使用 docker-create-image 目录中 README.md 镜像构建指引
```

## 启动命令
#### 基于 Transformer 模型的机器翻译任务
* 1、单卡模型训练
```shell
cd modle-train/transformer_train
 
# 修改训练配置文件
vim run_transformer_train.sh

# 训练模型 && 后台训练
bash run_transformer_train.sh
nohup bash run_transformer_train.sh > log.log 2>&1 &
```

* 2、单机多卡模型训练
```shell
cd modle-train/transformer_train
 
# 修改训练配置文件
vim run_transformer_train_ddp.sh

# 训练模型 && 后台训练
bash run_transformer_train_ddp.sh
nohup bash run_transformer_train_ddp.sh > log.log 2>&1 &
```

* 3、开启 test 测试服务 
```shell
cd test 
 
# 开启 ab_test 压测服务
bash test.sh
```


## 代码版本历史
* 1.0.0 - 2025/10/10
  * 创建 README.md 文件;
  * 构建项目整体目录架构;
  * 完善 Docker Container 容器构建 README.md;
  * 上传初版跑通的 model-train Transformer 机器翻译代码； 
* 1.0.1 - 2025/10/14
  * 梳理代码结构，初步完成工程化