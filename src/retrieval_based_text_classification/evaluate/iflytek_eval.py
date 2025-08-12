from retrieval_based_text_classification.model_factory import ModelFactory
from retrieval_based_text_classification.retriever.iflytek import IflytekRetriever
from loguru import logger
from datasets import load_dataset
from pathlib import Path
from traceback import format_exc


def load_retriever():
    """
    Load the IflytekRetriever with the specified model factory.
    """
    model_factory = ModelFactory(embedding_model_name="Qwen/Qwen3-Embedding-0.6B",
                                 reranker_model_name="Qwen/Qwen3-Reranker-0.6B")
    
    retriever = IflytekRetriever(
        uri="resources/scenes/iflytek/milvus.db",
        collection_name="iflytek",
        embedding_fn=model_factory.get_embedding_fn(),
        reranker_fn=model_factory.get_reranker_fn()
    )
    
    return model_factory, retriever

def load_val_dataset():
    """
    Load the validation dataset for Iflytek.
    """
    val = load_dataset("json", data_files={
        "val": "resources/scenes/iflytek/raw_data/dev.json"
    })["val"]
    
    return val

def log_retrive_result(result, batch, q_n=5, d_n=5, only_log_false=True):
    """
    Log the retrieval results.s
    
    q_n: Number of queries to log
         if -1, log all queries, else log the first q_n queries
    d_n: Number of documents to log per query
         if -1, log all documents, else log the first d_n documents
    only_log_false: If True, only log results where the top result does not match the label
    """
    logger.info("="*42 + "Retrieval Results" + "="*42)
    q_n = min(q_n, len(batch["sentence"])) if q_n != -1 else len(batch["sentence"])
    d_n = min(d_n, len(result[0])) if d_n != -1 else len(result[0])
    
    
    for q_idx in range(q_n):
        if only_log_false and result[q_idx][0]['label_des'] == batch['label_des'][q_idx]:
            continue
        
        logger.info(f"label: {batch['label_des'][q_idx]} | label_id: {batch['label'][q_idx]} | sentence: {batch['sentence'][q_idx]}")
        logger.info(f"Top-{len(result[q_idx])} retrieved results:")
        for item in result[q_idx][: d_n]:
            logger.info(f"rerank_score: {item['rerank_score']:.3f} | distance: {item['distance']:.3f} | label: {item['label_des']} | label_id: {item['label']} | sentence: {item['sentence']}")
        logger.info("-"*21 + f"Results for query {q_idx}" + '-'*21) if q_idx < q_n-1 else None


def eval(predicts, batch):
    """
    Evaluate the retrieval results.
    """
    correct = 0
    for q_idx in range(len(batch["sentence"])):
        if predicts[q_idx]['label_des'] == batch['label_des'][q_idx]:
            correct += 1
    
    accuracy = correct / len(batch["sentence"])
    logger.info("="*42 + "Evaluation Results" + "="*42)
    logger.info(f"Accuracy: {accuracy:.4f} ({correct}/{len(batch['sentence'])})")   
    
    

if __name__ == "__main__":
    """
    usage:
        cd retrieval-based-text-classification
        python -m retrieval_based_text_classification.evaluate.iflytek_eval
    """
    try:
        # 0. Config
        n = -1  # number of queries to test, can be adjusted as needed
               # -1 for all queries
        top_k = 5  # number of top results to retrieve
        
        q_n = -1  # number of queries to log
        d_n = -1  # number of documents to log per query
        only_log_false = True
        output_path = Path("outputs/2025-08-12_未训练Qwen3-embedding-0.6B/iflytek_eval.log")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(str(output_path))
        
        
        # 1. load retriever
        model_factory, retriever = load_retriever()
        logger.info("Retriever loaded successfully.")
        
        # 2. load validation dataset
        val = load_val_dataset()
        logger.info(f"Loaded validation data: {len(val)}.\n{val[0]}")
        
        # 3. start retrieval
        logger.info("Starting retrieval...")
        batch = val[:n]  
        queries = batch["sentence"]
        predicts, results = retriever.classify(query=queries, top_k=top_k, is_rerank=True, is_return_retrieve_result=True)
        
        # (可选) log the results
        log_retrive_result(results, batch, q_n=q_n, d_n=d_n, only_log_false=only_log_false)
        
        # 4. evaluate
        eval(predicts, batch)
        
         
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        logger.error(format_exc())
    finally:
        # clean
        model_factory.cleanup()