from retrieval_based_text_classification.model_factory import ModelFactory
from retrieval_based_text_classification.retriever.iflytek import IflytekRetriever
from loguru import logger
from datasets import load_dataset
from pathlib import Path
from traceback import format_exc
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import numpy as np


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
    Evaluate the retrieval results with detailed classification metrics.
    """
    # 提取真实标签和预测标签
    y_true = batch['label_des']  # 或者使用 batch['label'] 如果你想用数字标签
    y_pred = [predict['label_des'] for predict in predicts]
    
    # 计算准确率
    accuracy = accuracy_score(y_true, y_pred)
    correct_nums = accuracy_score(y_true, y_pred, normalize=False)
    
    # 生成详细的分类报告
    report = classification_report(y_true, y_pred, output_dict=False, zero_division=0)
    report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    
    # 生成混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    # 获取所有唯一的标签
    unique_labels = sorted(list(set(y_true + y_pred)))
    
    logger.info("="*50 + "Evaluation Results" + "="*50)
    logger.info(f"Overall Accuracy: {accuracy:.4f} ({correct_nums}/{len(y_true)})")
    logger.info(f"Detailed Classification Report:\n{report}")
    
    # 记录每个类别的详细指标
    logger.info("")
    logger.info("Per-class Metrics:")
    for label in unique_labels:
        if label in report_dict:
            metrics = report_dict[label]
            logger.info(f"Class '{label}': Precision={metrics['precision']:.4f}, "
                       f"Recall={metrics['recall']:.4f}, F1={metrics['f1-score']:.4f}, "
                       f"Support={metrics['support']}")
    
    # 记录宏平均和加权平均
    logger.info("")
    logger.info(f"Macro Average: Precision={report_dict['macro avg']['precision']:.4f}, "
               f"Recall={report_dict['macro avg']['recall']:.4f}, "
               f"F1={report_dict['macro avg']['f1-score']:.4f}")
    
    logger.info(f"Weighted Average: Precision={report_dict['weighted avg']['precision']:.4f}, "
               f"Recall={report_dict['weighted avg']['recall']:.4f}, "
               f"F1={report_dict['weighted avg']['f1-score']:.4f}")
    
    # 记录混淆矩阵（如果类别不太多的话）
    if len(unique_labels) <= 20:  # 只有当类别数量不太多时才显示混淆矩阵
        logger.info(f"\nConfusion Matrix:")
        logger.info(f"Labels: {unique_labels}")
        logger.info(f"\n{cm}")
    
    # 找出表现最好和最差的类别
    class_f1_scores = [(label, report_dict[label]['f1-score']) 
                       for label in unique_labels if label in report_dict]
    class_f1_scores.sort(key=lambda x: x[1], reverse=True)
    
    logger.info("")
    logger.info(f"Best performing classes (top 5):")
    for label, f1 in class_f1_scores[:5]:
        logger.info(f"  {label}: F1={f1:.4f}")
    
    logger.info("")
    logger.info(f"Worst performing classes (bottom 5):")
    for label, f1 in class_f1_scores[-5:]:
        logger.info(f"  {label}: F1={f1:.4f}")
    
    return {
        'accuracy': accuracy,
        'classification_report': report_dict,
        'confusion_matrix': cm,
        'unique_labels': unique_labels
    }


if __name__ == "__main__":
    """
    usage:
        cd retrieval-based-text-classification
        python -m retrieval_based_text_classification.evaluate.iflytek_eval
    """
    try:
        # 0. Config
        max_samples = -1  # 用于验证的样本数量，-1 for all queries
        is_rerank = True  # 是否使用reranker
        top_k = 5  # 用于reranker 的 top 数量
        
        q_n = -1  # number of queries to log
        d_n = -1  # number of documents to log per query

        is_return_retrieve_result = True  # 是否返回排序结果
        only_log_false = True  # 是否只 log 预测错误的样本
        
        exp_dir = "2025-08-12｜v3｜reranker-prompt｜acc"
        output_path = Path(f"outputs") / exp_dir / "iflytek_eval.log"
        
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
        batch = val[:max_samples]  
        queries = batch["sentence"]
        predicts, results = retriever.classify(query=queries, top_k=top_k, 
                                               is_rerank=is_rerank, 
                                               is_return_retrieve_result=is_return_retrieve_result)
        
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