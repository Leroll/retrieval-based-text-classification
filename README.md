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

该数据集共有1.7万多条关于app应用描述的长文本标注数据，包含和日常生活相关的各类应用主题，共119个类别："打车":0,"地图导航":1,"免费WIFI":2,"租车":3,….,"女性":115,"经营":116,"收款":117,"其他":118(分别用0-118表示)。  

数据地址: https://github.com/CLUEbenchmark/CLUE  
C_MTEB地址: https://github.com/FlagOpen/FlagEmbedding/tree/master/research/C_MTEB  

原始数据下载:
```bash
cd resources/data
wget https://storage.googleapis.com/cluebenchmark/tasks/iflytek_public.zip
```




# TODO 
- [ ] 搭建完整pipeline
    - [x] 完成基于qwen3 embedding & reranker 的模型模块，切采用vllm版