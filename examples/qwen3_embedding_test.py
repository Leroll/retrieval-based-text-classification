import os
from dotenv import load_dotenv
load_dotenv()
from loguru import logger
from pathlib import Path

def setup_cache_paths():
    """set up cache paths for HF and ModelScope
    """
    project_root = Path(__file__).parent.parent
    
    hf_hub_cache = os.environ.get('HF_HUB_CACHE', 'checkpoints/hf_cache')
    modelscope_cache = os.environ.get('MODELSCOPE_CACHE', 'checkpoints/modelscope_cache')
    
    hf_hub_cache = Path(hf_hub_cache)
    modelscope_cache = Path(modelscope_cache)
    
    if not hf_hub_cache.is_absolute():
        hf_hub_cache = project_root / hf_hub_cache
    if not modelscope_cache.is_absolute():
        modelscope_cache = project_root / modelscope_cache
    
    # set absolute paths to environment variables
    os.environ['HF_HUB_CACHE'] = str(hf_hub_cache)
    os.environ['MODELSCOPE_CACHE'] = str(modelscope_cache)
    
    hf_hub_cache.mkdir(parents=True, exist_ok=True)
    modelscope_cache.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"HF_HUB_CACHE set to: {hf_hub_cache}")
    logger.info(f"MODELSCOPE_CACHE set to: {modelscope_cache}")
    
    return hf_hub_cache, modelscope_cache

setup_cache_paths()

#-----------------------------------
# Embedding
#-----------------------------------
embedding_model_name = "Qwen/Qwen3-Embedding-0.6B"

# Requires vllm>=0.8.5
import torch
import vllm
from vllm import LLM
from vllm.distributed.parallel_state import destroy_model_parallel

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery:{query}'

# Each query must come with a one-sentence instruction that describes the task
task = 'Given a web search query, retrieve relevant passages that answer the query'

queries = [
    get_detailed_instruct(task, 'What is the capital of China?'),
    get_detailed_instruct(task, 'Explain gravity')
]
# No need to add instruction for retrieval documents
documents = [
    "The capital of China is Beijing.",
    "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun."
]
input_texts = queries + documents

model = LLM(model=embedding_model_name, task="embed")

outputs = model.embed(input_texts)
embeddings = torch.tensor([o.outputs.embedding for o in outputs])
scores = (embeddings[:2] @ embeddings[2:].T)
print(scores.tolist())
# [[0.7620252966880798, 0.14078938961029053], [0.1358368694782257, 0.6013815999031067]]

# 清理资源
destroy_model_parallel()

# 如果仍有警告，可以添加以下清理代码
import torch.distributed as dist
try:
    if dist.is_initialized():
        dist.destroy_process_group()
except Exception as e:
    logger.debug(f"Process group cleanup: {e}")

# 清理CUDA缓存
torch.cuda.empty_cache()

