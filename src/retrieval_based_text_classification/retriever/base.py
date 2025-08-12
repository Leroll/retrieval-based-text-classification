from abc import ABC, abstractmethod
from pymilvus import MilvusClient
from typing import List, Dict
from loguru import logger
from pathlib import Path    


class BaseRetriever(ABC):
    
    def __init__(self, uri: str, collection_name: str, embedding_fn, reranker_fn=None):
        """
        Args:
            uri: Lite版本传文件路径，Server版本传服务器地址
            collection_name: 集合名称
        """        
        self.client = MilvusClient(uri=uri)
        self.schema = self._get_schema()
        self.index_params = self._get_index_params()
        self.collection_name = collection_name
        if not self.client.has_collection(collection_name):
            self._create_collection()
        
        self.embedding_fn = embedding_fn
        self.reranker_fn = reranker_fn
    
    @abstractmethod
    def _get_schema(self):
        """创建schema
        """
        pass
    
    @abstractmethod 
    def _get_index_params(self):
        """创建索引参数
        
        Milvus Lite 仅支持FLAT索引类型。无论在 Collections 中指定了哪种索引类型，它都使用 FLAT 类型。
        """
        pass
    
    def _create_collection(self):
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=self.schema,
            index_params=self.index_params
        )
    
    def recreate_collection(self):
        """
        重新创建集合（删除旧集合，创建新集合）
        
        在schema或者index变更时使用
        """
        if self.client.has_collection(self.collection_name):
            logger.info(f"Dropping existing collection: {self.collection_name}")
            self.client.drop_collection(self.collection_name)
        
        logger.info(f"Creating new collection: {self.collection_name}")
        self._create_collection()
    
    @abstractmethod
    def batch_insert(self, data: List[Dict]):
        """
        批量插入数据
        Args:
            data: 批量数据，每个元素是一个字典，包含字段名和对应的值
        """
        pass
    
    @abstractmethod
    def file_insert(self, file_path: Path):
        """从文件中批量导入数据，
        
        读取数据，调用batch_insert方法插入数据
        """
        pass
    
    @abstractmethod
    def _retrieve(self, query: List[str], 
                  top_k: int = 5, filter_str: str = None, is_rerank: bool = False) -> List[Dict]:
        """检索并返回检索结果
        """
        pass
    
    @abstractmethod
    def classify(self, query: List[str], 
                  top_k: int = 5, filter_str: str = None, is_rerank: bool = False, 
                  max_batch_size: int = 32, is_return_retrieve_result: bool = False) -> List[Dict]:
        """分类查询
        
        Args:
            query: 查询语句列表
            top_k: 返回的结果数量
            filter_str: 过滤条件
            is_rerank: 是否进行重排序
            max_batch_size: 最大批处理大小
            is_return_retrieve_result: 是否返回检索结果
        """
        pass
        
        