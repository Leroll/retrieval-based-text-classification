from pymilvus import MilvusClient, DataType
from typing import List
from loguru import logger
from pydantic import BaseModel, computed_field
from pathlib import Path
from datasets import load_dataset
from typing import Dict
import hashlib
import json

from .base import BaseRetriever

class IflytekData(BaseModel):
    """定义Iflytek数据模型
    """
    label: int
    label_des: str
    sentence: str
    embedding: List[float] = None  # 用于存储嵌入向量    
    
    @computed_field
    def data_id(self) -> str:
        """主键, 使用sentence的hash值计算，作为数据唯一标识
        """
        return self.calculate_hash(self.sentence)
    
    @computed_field
    def sentence_len(self) -> int:
        """原始句子长度
        """
        return len(self.sentence) if self.sentence else 0
        
    @staticmethod
    def calculate_hash(sentence: str) -> str:
        """
        计算句子的hash值作为ID
        使用MD5哈希并返回十六进制字符串
        """
        hash_obj = hashlib.md5(sentence.encode('utf-8'))
        return hash_obj.hexdigest()
        
    @classmethod
    def get_milvus_schema(cls):
        """
        获取Milvus集合的Schema, 保持和上述字段一致
        
        Returns:
            schema: Milvus集合的Schema
        """
        schema = MilvusClient.create_schema()
        schema.add_field(field_name="data_id", datatype=DataType.VARCHAR, is_primary=True, max_length=64, 
                         description="数据唯一标识, hash of sentence")
        schema.add_field(field_name="label", datatype=DataType.INT8, 
                         description="分类标签id")
        schema.add_field(field_name="label_des", datatype=DataType.VARCHAR, max_length=64, 
                         description="分类标签")
        schema.add_field(field_name="sentence", datatype=DataType.VARCHAR, max_length=640, 
                         description="app描述文本")  # 依据数据长度分布得到
        schema.add_field(field_name="sentence_len", datatype=DataType.INT16,
                         description="原始句子长度")
        schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=768,
                         description="sentence embedding")
        
        return schema
        
class IflytekRetriever(BaseRetriever):
    
    def __init__(self, uri: str, collection_name: str,
                 embedding_fn=None, reranker_fn=None):
        """
        """
        super().__init__(uri, collection_name, embedding_fn, reranker_fn)
        
        self.reranker_prompt = "下面的<Query>和<Document>分别是两个app的应用描述，判断它们是否属于同一分类。"
    
    def _get_schema(self):
        return IflytekData.get_milvus_schema()
    
    def _get_index_params(self):
        """创建索引参数
        """
        index_params = MilvusClient.prepare_index_params()

        # lite 模式不支持
        # index_params.add_index(
        #     field_name="embedding", 
        #     index_type="HNSW_SQ", 
        #     index_name="embedding_index",
        #     metric_type="COSINE", 
        #     params={
        #         "M": 64, # 较大的M 通常会提高准确率，但会增加内存开销，并减慢索引构建和搜索速度。建议值[5, 100].
        #         "efConstruction": 100, # efConstruction 越高，索引越准确，因为会探索更多潜在连接。建议值[50, 500].
        #         "sq_type": "SQ8", 
        #         "refine": True,
        #         "refine_type": "FP32"
        #     } 
        # )

        index_params.add_index(
            field_name="embedding", 
            index_type="IVF_FLAT", 
            index_name="embedding_index", 
            metric_type="COSINE", 
            params={
                "nlist": 128, # nlist 值越大，通过创建更精细的簇来提高召回率，但会增加索引构建时间。
                             # 建议值 [32, 4096]
            }
        )

        return index_params
    
    def batch_insert(self, data: List[Dict], batch_size: int = 1024):
        """
        批量插入数据到Iflytek的Milvus集合中
        Args:
            data: 批量数据
                {
                    'label': '11',
                    'label_des': '薅羊毛',
                    'sentence': "xxx"
                }
            batch_size: 每批次处理的数据量
        """
        if not data:
            logger.warning("No data to insert.")
            return
        
        total_batches = (len(data) + batch_size - 1) // batch_size
        success_count = 0
        
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            batch_num = i // batch_size + 1
            logger.info(f"Processing batch {batch_num}/{total_batches} with {len(batch)} items.")
            
            batch = [IflytekData(**item) for item in batch]
            
            # 为批次中的每个项目生成嵌入向量
            items_need_embedding = [item for item in batch if item.embedding is None]
            if items_need_embedding:
                sentences = [item.sentence for item in items_need_embedding]
                embeddings = self.embedding_fn(sentences, task="")
                
                # 将生成的嵌入向量分配给对应的项目
                for item, embedding in zip(items_need_embedding, embeddings):
                    item.embedding = embedding
                logger.info(f"Generated embeddings for {len(items_need_embedding)} items in batch {batch_num}.")
            
            # logger.info(f"Item example:\n{batch[0].model_dump()}")
            
            # 插入当前批次数据
            res = self.client.upsert(
                collection_name=self.collection_name,
                data=[item.model_dump() for item in batch]
            )
            logger.info(f"Batch {batch_num} insert result: {res}")
            success_count += res.get('upsert_count', 0)
        
        logger.info(f"Total data num: {len(data)} | successfully inserted: {success_count} | collection: {self.collection_name}.")
        
    def _retrieve(self, query: List[str], 
                 top_k: int = 5, filter_str: str = None, is_rerank: bool = False):
        
        embeddings = self.embedding_fn(query, task="")
        
        # 1. milvus 相似度检索
        search_params = {
            "params": {
                "nprobe": 10, # Number of clusters to search
            }
        }

        result = self.client.search(
            collection_name=self.collection_name, # Collection name
            data=embeddings,
            filter=filter_str,
            limit=top_k, 
            output_fields=["data_id", "label", "label_des", "sentence"],
            search_params=search_params,
            anns_field="embedding",
        )
        
        # 2. 如果需要，使用reranker进行二次排序
        if is_rerank and self.reranker_fn:
            
            for q_idx in range(len(query)):
                pairs = [ [query[q_idx], d]  for d in result[q_idx] ]
                scores = self.reranker_fn(pairs, task=self.reranker_prompt)
                for d_idx, doc in enumerate(result[q_idx]):
                    doc['rerank_score'] = scores[d_idx]
                result[q_idx].sort(key=lambda x: x['rerank_score'], reverse=True)
            
        return result
    
    def classify(self, query: List[str], 
                 top_k: int = 5, filter_str: str = None, is_rerank: bool = False, 
                 max_batch_size: int = 32, 
                 is_return_retrieve_result: bool = False,):
        """
        分类查询
        
        is_return_retrieve_result: 是否返回检索结果
        """
        if not query:
            logger.warning("No query provided.")
            return []
        
        
        all_predicts = []
        all_results = []
        for i in range(0, len(query), max_batch_size):
            batch_query = query[i:i + max_batch_size]
            logger.info(f"Processing batch {i // max_batch_size + 1} with {len(batch_query)} queries.")
            
            # 调用检索方法
            result = self._retrieve(batch_query, top_k, filter_str, is_rerank)
            predicts = [item[0] for item in result]
            
            all_predicts.extend(predicts)
            all_results.extend(result) if is_return_retrieve_result else None

        return (all_predicts, all_results) if is_return_retrieve_result else all_predicts
        

class DataImporter:
    """
    数据导入工具类
    
    记录各批次数据导入
    """
    def import_20250811_iflytek_train(self, retriever: IflytekRetriever):
        # 2025-08-11 - iflytek train.json 导入
        data_path = Path("resources/scenes/iflytek/raw_data/train.json")
        logger.info(f"Importing... | 2025-08-11 | {data_path.name}")
        data = load_dataset("json", data_files={"train": str(data_path) })['train']
        retriever.batch_insert(data.to_list(), batch_size=1024)
        
    
    def run(self, is_recreate: bool = False):
        """ 
        """
        # 加载model, 不同批次的导入保证 embedding model 一致
        from retrieval_based_text_classification.model_factory import ModelFactory
        model_factory = ModelFactory(embedding_model_name="Qwen/Qwen3-Embedding-0.6B",
                                    reranker_model_name=None)
        
        # 初始化retriever
        retriever = IflytekRetriever(
            uri="resources/scenes/iflytek/milvus.db",
            collection_name="iflytek",
            embedding_fn=model_factory.get_embedding_fn(),
            reranker_fn=model_factory.get_reranker_fn()
        )
        if is_recreate:
            retriever.recreate_collection()
        
        logger.info("-"*42)
        logger.info("Initialized IflytekRetriever.")
        logger.info(f"collection name: {retriever.collection_name}")
        logger.info(f"schema:\n{json.dumps(retriever.schema.to_dict(), indent=4, ensure_ascii=False)}")
        logger.info(f"index params:\n{retriever.index_params}")
        logger.info(f"state:\n{retriever.client.get_collection_stats(collection_name=retriever.collection_name)}")
        logger.info(f"load state:\n{retriever.client.get_load_state(retriever.collection_name)}")
        logger.info("-"*42)
        
        # 开始导入数据
        logger.info("Press any key to start importing data...")
        input()
        self.import_20250811_iflytek_train(model_factory, retriever)
        
        
        # flush数据
        retriever.client.flush(retriever.collection_name)
        logger.info("Data flushed to Milvus.")
        
        
        # 插入完毕
        logger.info("-"*42)
        logger.info("Data import completed.")
        logger.info(f"collection row count: {retriever.client.get_collection_stats(collection_name=retriever.collection_name)['row_count']}")
        logger.info("Sample data from collection:")
        sample_data = retriever.client.query(collection_name=retriever.collection_name, limit=5)
        for item in sample_data:
            item['embedding'] = item['embedding'][:10]  # 避免打印过长的嵌入向量
            logger.info(item)
        logger.info("-"*42)
        logger.info(f"collection state:\n{retriever.client.get_collection_stats(collection_name=retriever.collection_name)}")

if __name__ == "__main__":
    """
    usage:
        cd retrieval-based-text-classification
        python -m retrieval_based_text_classification.retriever.iflytek
    """
    data_importer = DataImporter()
    data_importer.run(is_recreate=True)



