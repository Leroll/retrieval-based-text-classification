# embeddings 
from dotenv import load_dotenv
load_dotenv()

import torch
from vllm import LLM, SamplingParams, PoolingParams
from vllm.distributed.parallel_state import destroy_model_parallel
import gc
import math
from vllm.inputs.data import TokensPrompt
from transformers import AutoTokenizer
from typing import List, Tuple
from loguru import logger
import numpy as np
from pathlib import Path
import os 
import torch.distributed as dist

from retrieval_based_text_classification.utils import Singleton

@Singleton
class ModelFactory:
    """模型工厂类，用于初始化和管理embedding和rerank模型
    
    Qwen3-embediing: Requires vllm>=0.8.5
    """
    def __init__(self, embedding_model_name: str = "Qwen/Qwen3-Embedding-0.6B", 
                 reranker_model_name: str = "Qwen/Qwen3-Reranker-0.6B"):
        self.embedding_model_name = embedding_model_name 
        self.reranker_model_name = reranker_model_name 
        
        self.hf_cache_path, self.modelscope_cache_path = self._setup_cache_paths()
        
        # 添加模型实例变量
        self.embedding_model = None
        self.reranker_model = None
        self.tokenizer = None
        
        self.number_of_gpu = torch.cuda.device_count()
        logger.info(f"Initialized ModelFactory with {self.number_of_gpu} GPUs")
        
        # 初始化函数
        self.embedding_fn = self._initialize_embedding_fn() if self.embedding_model_name else None
        self.reranker_fn = self._initialize_reranker_fn() if self.reranker_model_name else None
        
    def _setup_cache_paths(self):
        """set up cache paths for HF and ModelScope
        """
        project_root = Path(__file__).resolve().parent.parent.parent
        
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
        
    #-----------------------------------
    # Embedding Functionality
    #-----------------------------------
    def _initialize_embedding_fn(self):
        """初始化embedding_fn
        
        embedding_fn 
            输入: List[str] - 查询列表
            输出: List[List[float]] - embeddings
        """
        logger.info("Initializing embedding model...")
        
        if "Qwen3-Embedding" in self.embedding_model_name:
            embedding_fn = self._get_qwen3_embedding_fn()
        else:
            raise ValueError(f"Unsupported model name: {self.embedding_model_name}")
        
        logger.info("Embedding model initialized successfully")
        
        return embedding_fn
        
    def _get_detailed_instruct(self, task_description: str, query: str) -> str:
        """为查询添加任务描述
        """
        return f'Instruct: {task_description}\nQuery:{query}'
    
    def _get_qwen3_embedding_fn(self):
        """
        model_name = Qwen/Qwen3-Embedding-0.6B
        """
        self.embedding_model = LLM(model=self.embedding_model_name, 
                                   task="embed", 
                                   hf_overrides={"is_matryoshka": True})  # Matryoshka Representation Learning, 支持多种维度
        default_task = 'Given a web search query, retrieve relevant passages that answer the query'
        
        def embedding_fn(queries: List[str], task: str = None) -> List[List[float]]:
            """
            生成embedding向量
            
            Args:
                queries: 查询列表
                task: 任务描述
                
            Returns:
                List[List[float]]: embeddings列表，每个元素是一个向量
            """
            if task is None:
                task = default_task
            
            formatted_queries = [self._get_detailed_instruct(task, query) for query in queries]
            outputs = self.embedding_model.embed(formatted_queries, 
                                                 pooling_params=PoolingParams(dimensions=768))  # 使用768维度的嵌入
            embeddings = [o.outputs.embedding for o in outputs]
            
            return embeddings
        
        return embedding_fn
    
    def get_embedding_fn(self):
        """获取embedding函数
        """
        return self.embedding_fn
    
    
    #-----------------------------------
    # Reranker Functionality
    #-----------------------------------
    def _initialize_reranker_fn(self):
        """初始化reranker_fn
    
        reranking_fn:
            输入: List[str] - 查询列表
                 List[str] - 文档列表
            输出: List[float] - 每个查询-文档对的相关性分数
        """
        logger.info("Initializing reranker model...")
        if "Qwen3-Reranker" in self.reranker_model_name:
            reranker_fn = self._get_qwen3_reranker_fn()
        else:
            raise ValueError(f"Unsupported model name: {self.reranker_model_name}")
        
        logger.info("Reranker model initialized successfully")
        
        return reranker_fn
        
    def _format_instruction(self, 
                            instruction: str, 
                            query: str, 
                            doc: str) -> List[dict]:
        """格式化rerank指令
        """
        text = [
            {"role": "system", "content": "Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."},
            {"role": "user", "content": f"<Instruct>: {instruction}\n\n<Query>: {query}\n\n<Document>: {doc}"}
        ]
        return text
        
    def _process_inputs(self, 
                        pairs: List[Tuple[str, str]], 
                        instruction: str, 
                        max_length: int, 
                        suffix_tokens: List[int]) -> List[TokensPrompt]:
        """处理输入数据
        """
        messages = [ self._format_instruction(instruction, query, doc) for query, doc in pairs ]
        messages = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False, enable_thinking=False
        )
        messages = [ ele[:max_length] + suffix_tokens for ele in messages ]
        messages = [ TokensPrompt(prompt_token_ids=ele) for ele in messages ]
        return messages
    
    def _compute_logits(self, 
                        messages: List[TokensPrompt], 
                        sampling_params: SamplingParams, 
                        true_token: int, 
                        false_token: int) -> List[float]:
        """计算logits并转换为分数
        """
        outputs = self.reranker_model.generate(messages, sampling_params, use_tqdm=False)
        scores = []
        
        for i in range(len(outputs)):
            final_logits = outputs[i].outputs[0].logprobs[-1]
            
            if true_token not in final_logits:
                true_logit = -10
            else:
                true_logit = final_logits[true_token].logprob
                
            if false_token not in final_logits:
                false_logit = -10
            else:
                false_logit = final_logits[false_token].logprob
                
            true_score = math.exp(true_logit)
            false_score = math.exp(false_logit)
            score = true_score / (true_score + false_score)
            scores.append(score)
            
        return scores

    def _get_qwen3_reranker_fn(self):
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.reranker_model_name)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 初始化reranking模型
        self.reranker_model = LLM(
            model=self.reranker_model_name, 
            tensor_parallel_size=self.number_of_gpu, 
            max_model_len=5120,   # 默认 10000
            enable_prefix_caching=True, 
            gpu_memory_utilization=0.6
        )
        
        # 配置参数
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        max_length = 1024  # 默认 8192
        suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)
        true_token = self.tokenizer("yes", add_special_tokens=False).input_ids[0]
        false_token = self.tokenizer("no", add_special_tokens=False).input_ids[0]
        
        sampling_params = SamplingParams(
            temperature=0, 
            max_tokens=1,
            logprobs=20, 
            allowed_token_ids=[true_token, false_token],
        )
        
        default_task = 'Given a web search query, retrieve relevant passages that answer the query'
        
        def reranker_fn(pairs: List[Tuple[str, str]], 
                      task: str = None) -> List[float]:
            """
            对查询-文档对进行重排序评分
            
            Args:
                pairs: 查询-文档对列表，每个元素为 (query, document) 元组
                task: 任务描述
                
            Returns:
                List[float]: 每个查询-文档对的相关性分数
            """
            
            if task is None:
                task = default_task
            
            # 处理输入
            inputs = self._process_inputs(pairs, task, max_length - len(suffix_tokens), suffix_tokens)
            
            # 计算分数
            scores = self._compute_logits(inputs, sampling_params, true_token, false_token)
            
            return scores
        
        return reranker_fn
    
    
    def get_reranker_fn(self):
        """获取reranker函数
        """
        return self.reranker_fn

    #-----------------------------------
    # 其他功能
    #-----------------------------------
    def cleanup(self):
        """清理资源"""
        logger.info("Cleaning up model resources...")
        
        if self.embedding_model is not None:
            del self.embedding_model
            self.embedding_model = None
        if self.reranker_model is not None:
            del self.reranker_model
            self.reranker_model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()
        gc.collect()
        torch.cuda.empty_cache()
    
        logger.info("Model resources cleaned up successfully")
    
    def __del__(self):
        """析构函数，在对象被垃圾回收时自动调用cleanup"""
        try:
            self.cleanup()
        except Exception as e:
            # 在析构函数中避免抛出异常
            logger.warning(f"Error during ModelFactory destruction: {e}")


# 使用示例
if __name__ == "__main__":
    try:
        # 初始化模型工厂
        logger.info("Initializing ModelFactory...")
        model_factory = ModelFactory()
        
        # 测试数据
        task = 'Given a web search query, retrieve relevant passages that answer the query'
        queries = ["What is the capital of China?", "Explain gravity"]
        documents = [
            "The capital of China is Beijing, which is also the political and cultural center of the country.",
            "Gravity is a fundamental force that attracts objects with mass toward each other. It keeps planets in orbit around stars.",
            "Python is a programming language that is widely used for data science and machine learning applications.",
            "The weather today is sunny with temperatures reaching 25 degrees Celsius."
        ]
        
        # 测试embedding功能
        logger.info("Testing embedding function...")
        try:
            embeddings = model_factory.embedding_fn(queries, task)
            logger.info(f'Generated embeddings shape: {embeddings.shape}')
            logger.info(f'First query embedding (first 5 dims): {embeddings[0][:5].tolist()}')
        except Exception as e:
            logger.error(f"Embedding test failed: {e}")
        
        # 测试批量rerank
        logger.info("Testing batch rerank...")
        try:
            # 创建查询-文档对进行批量测试
            query_doc_pairs = []
            for query in queries:
                for doc in documents:
                    query_doc_pairs.append((query, doc))
            
            # 使用新的pairs格式进行批量rerank
            batch_scores = model_factory.reranker_fn(query_doc_pairs, task)
            logger.info(f'Batch rerank generated {len(batch_scores)} scores')
            logger.info(f'Sample batch scores: {batch_scores[:4]}')
            
            # 展示批量结果
            score_idx = 0
            for i, query in enumerate(queries):
                logger.info(f"Query {i+1}: '{query}'")
                for j, doc in enumerate(documents):
                    logger.info(f"  Doc {j+1}: {batch_scores[score_idx]:.4f} - {doc[:50]}...")
                    score_idx += 1
                    
        except Exception as e:
            logger.error(f"Batch rerank test failed: {e}")
            
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
    finally:
        # 清理资源
        logger.info("Cleaning up resources...")
        try:
            model_factory.cleanup()
            logger.info("Resources cleaned up successfully")
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")