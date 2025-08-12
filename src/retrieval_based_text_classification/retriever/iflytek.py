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
                embeddings = self.embedding_fn(sentences)
                
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
    
    def file_insert(self, file_path: Path = "resources/scenes/iflytek/raw_data/train.json"):
        """
        从文件中批量插入数据
        Args:
            file_path: 文件路径，文件内容为JSON格式的列表
            
        sample:
            {
                'label': '11',
                'label_des': '薅羊毛',
                'sentence': "xxx"
            }
        """
        data = load_dataset("json", data_files={"train": str(file_path) })['train']
        self.batch_insert(data.to_list(), batch_size=1024)
        

class DataImporter:
    """
    数据导入工具类
    
    记录各批次数据导入
    """
    def import_20250811_iflytek_train(self, model_factory, retriever: IflytekRetriever):
        # 2025-08-11 - iflytek train.json 导入
        
        logger.info("Importing 2025-08-11 iflytek train.json data...")
        retriever.file_insert(file_path=Path("resources/scenes/iflytek/raw_data/train.json"))
        
    
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



