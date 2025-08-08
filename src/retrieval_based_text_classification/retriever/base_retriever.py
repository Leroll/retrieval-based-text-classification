from abc import ABC, abstractmethod
from pymilvus import MilvusClient
from pymilvus.bulk_writer import LocalBulkWriter, bulk_importer
from typing import List, Dict, Optional
from loguru import logger

class BaseRetriever(ABC):
    
    def __init__(self, uri: str, collection_name: str, embedding_fn, reranker_fn=None):
        """
        Args:
            uri: Lite版本传文件路径，Server版本传服务器地址
            collection_name: 集合名称
        """        
        self.client = MilvusClient(uri=uri)
        self.schema = self._get_schema()
        self.collection_name = collection_name
        if not self.client.has_collection(collection_name):
            self._create_collection()
        
        self.embedding_fn = embedding_fn
        self.reranker_fn = reranker_fn
    
    @abstractmethod
    def _get_schema(self):
        pass
    
    def _create_collection(self):
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=self.schema
        )
    
    @abstractmethod
    def batch_insert(self, data: List[Dict]):
        """
        批量插入数据
        Args:
            data: 批量数据，每个元素是一个字典，包含字段名和对应的值
        """
        pass
    
    