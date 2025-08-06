# retrieval-based-text-classification


# 快速开始
```bash
# 环境准备
uv venv --python 3.10 --seed
source .venv/bin/activate
uv sync
cp .env.example .env

# 启动服务
rtc serve
```


# 数据
选用 **IFlyTek** 长文本分类数据集作为样例构建系统，它属于MTEB中文任务中的一个子任务。  
数据地址: https://github.com/CLUEbenchmark/CLUE  
C_MTEB地址: https://github.com/FlagOpen/FlagEmbedding/tree/master/research/C_MTEB  

原始数据下载:
```bash
cd resources/data
wget https://storage.googleapis.com/cluebenchmark/tasks/iflytek_public.zip
```




# TODO 
- [ ] 搭建完整pipeline