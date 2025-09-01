#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
新闻文本语义相似度演示脚本
专门展示如何处理新闻文本中关键词与文本语义相近但不完全匹配的情况
"""
from keyword_similarity import KeywordTextSimilarity
import time


def news_semantic_similarity_demo():
    """
    新闻文本语义相似度演示
    展示如何处理关键词与文本语义相近但不完全匹配的情况
    """
    print("=" * 80)
    print("          新闻文本语义相似度演示 - 处理关键词与文本不完全匹配的情况")
    print("=" * 80)
    print("此演示将展示如何在新闻文本中找到与查询关键词语义相关但可能不完全匹配的内容")
    print("\n")
    
    # 创建相似度计算实例
    similarity = KeywordTextSimilarity()
    
    # 准备新闻文档集 - 更丰富的新闻示例
    news_documents = [
        (0, "国家航天局宣布，我国首次火星探测任务天问一号探测器成功着陆火星乌托邦平原南部预选着陆区，迈出了我国星际探测征程的重要一步。这是中国航天事业发展的又一具有里程碑意义的进展。"),
        (1, "国务院总理主持召开国务院常务会议，部署进一步支持小微企业、个体工商户纾困和发展的措施，确定加强农村义务教育薄弱环节的工作举措。会议指出，小微企业和个体工商户是我国市场主体的重要组成部分。"),
        (2, "世界卫生组织发布最新研究报告，强调全球各国应加强合作，共同应对气候变化对公共卫生带来的挑战。报告指出，气候变化可能导致多种疾病的传播范围扩大。"),
        (3, "国际能源署发布最新预测，全球清洁能源投资在2023年首次超过化石燃料投资，标志着能源转型进程加速。可再生能源如太阳能、风能等在全球能源结构中的占比持续提升。"),
        (4, "国家发改委等部门联合发布通知，部署加快推动新型储能发展，助力构建新型电力系统。通知明确了未来五年新型储能发展的主要目标和重点任务。"),
        (5, "教育部发布2023年全国教育事业发展统计公报，数据显示全国共有各级各类学校51.85万所，在校生2.93亿人，专任教师1880.36万人，教育普及水平持续提升。"),
        (6, "中国科学院宣布，在量子计算领域取得重大突破，成功构建76个光子的量子计算原型机'九章二号'，求解特定问题比目前最快的超级计算机快亿亿亿倍。"),
        (7, "国家统计局发布数据，2023年我国国内生产总值(GDP)超过126万亿元，按不变价格计算，比上年增长5.2%，经济运行总体回升向好，高质量发展扎实推进。"),
        (8, "外交部发言人表示，中方一贯坚持在相互尊重、平等互利的基础上发展同各国的友好合作关系，愿同国际社会一道，共同维护世界和平与发展。"),
        (9, "国家医保局发布消息，新一轮医保药品目录调整工作正式启动，将进一步提高医保药品保障水平，减轻群众用药负担。")
    ]
    
    # 添加文档到相似度计算实例
    print("正在添加新闻文档...")
    for doc_id, doc_text in news_documents:
        similarity.add_document(doc_id, doc_text)
    print(f"已添加{len(news_documents)}个新闻文档。\n")
    
    # 定义专门测试语义相似度的查询
    # 这些查询中的关键词与文档内容不完全匹配，但在语义上是相关的
    semantic_queries = [
        {
            "query": "太空探索 宇宙任务",
            "relevant_docs": [0],
            "description": "查询词与文档0语义相关但没有完全匹配的关键词"
        },
        {
            "query": "小型企业 扶持政策",
            "relevant_docs": [1],
            "description": "查询词与文档1语义相关但表述不同"
        },
        {
            "query": "全球变暖 健康问题",
            "relevant_docs": [2],
            "description": "查询词是文档2中概念的同义表述"
        },
        {
            "query": "可持续能源 绿色转型",
            "relevant_docs": [3],
            "description": "查询词与文档3讨论的主题高度相关但用词不同"
        },
        {
            "query": "量子科技 技术革新",
            "relevant_docs": [6],
            "description": "查询词概括了文档6的核心内容但没有使用完全相同的词汇"
        }
    ]
    
    # 加载深度学习模型
    print("\n" + "-" * 80)
    print("正在加载深度学习模型...")
    try:
        start_time = time.time()
        # 可以尝试使用中文优化的模型
        # similarity.load_sentence_transformer('uer/sbert-base-chinese-nli')
        similarity.load_sentence_transformer()  # 默认使用'all-MiniLM-L6-v2'
        load_time = time.time() - start_time
        print(f"模型加载完成，耗时: {load_time:.2f}秒")
        model_loaded = True
    except Exception as e:
        print(f"模型加载失败: {str(e)}")
        print("请确保已安装所需依赖: pip install transformers torch sentence-transformers")
        model_loaded = False
    print("-" * 80 + "\n")
    
    # 对每个查询进行测试
    for item in semantic_queries:
        query = item["query"]
        relevant_docs = item["relevant_docs"]
        description = item["description"]
        
        print(f"\n\n========== 查询关键词: '{query}' ==========")
        print(f"说明: {description}")
        print(f"理论上最相关的文档ID: {relevant_docs}")
        
        # 使用传统方法 - TF-IDF
        print("\n--- TF-IDF 结果 (传统方法) ---")
        tf_idf_results = similarity.tf_idf_similarity(query)
        for doc_id, doc_text, score in tf_idf_results[:3]:
            relevance = "✓ 相关" if doc_id in relevant_docs else "✗ 不相关"
            print(f"文档{doc_id} (得分: {score:.4f}) [{relevance}]: {doc_text[:60]}...")
        
        # 使用传统方法 - BM25
        print("\n--- BM25 结果 (传统方法) ---")
        bm25_results = similarity.bm25_similarity(query)
        for doc_id, doc_text, score in bm25_results[:3]:
            relevance = "✓ 相关" if doc_id in relevant_docs else "✗ 不相关"
            print(f"文档{doc_id} (得分: {score:.4f}) [{relevance}]: {doc_text[:60]}...")
        
        # 使用深度学习方法 - Sentence-BERT
        if model_loaded:
            print("\n--- Sentence-BERT 结果 (深度学习语义方法) ---")
            sentence_bert_results =  similarity.sentence_transformer_similarity(query)
            for doc_id, doc_text, score in sentence_bert_results[:3]:
                relevance = "✓ 相关" if doc_id in relevant_docs else "✗ 不相关"
                print(f"文档{doc_id} (得分: {score:.4f}) [{relevance}]: {doc_text[:60]}...")
            
            # 使用增强语义相似度方法 (混合BM25和Sentence-BERT)
            print("\n--- 增强语义相似度 结果 (混合方法) ---")
            # 给语义相似度更高的权重 (0.3表示BM25的权重，0.7表示语义相似度的权重)
            enhanced_results = similarity.semantic_enhanced_similarity(query, method='hybrid', weights=(0.3, 0.7))
            for doc_id, doc_text, score in enhanced_results[:3]:
                relevance = "✓ 相关" if doc_id in relevant_docs else "✗ 不相关"
                print(f"文档{doc_id} (得分: {score:.4f}) [{relevance}]: {doc_text[:60]}...")
        
        print("\n" + "-" * 80)
    
    # 展示不同权重组合的效果
    if model_loaded:
        print("\n\n" + "=" * 80)
        print("           不同权重组合对结果的影响演示")
        print("=" * 80)
        
        query = "太空探索 宇宙任务"  # 使用第一个查询作为示例
        print(f"查询关键词: '{query}'\n")
        
        # 测试不同的权重组合
        weight_combinations = [
            (0.9, 0.1),  # 几乎完全依赖BM25
            (0.7, 0.3),  # 偏向BM25
            (0.5, 0.5),  # 平衡
            (0.3, 0.7),  # 偏向语义相似度（推荐）
            (0.1, 0.9)   # 几乎完全依赖语义相似度
        ]
        
        for weights in weight_combinations:
            bm25_weight, semantic_weight = weights
            print(f"\n--- 权重组合: BM25={bm25_weight}, 语义相似度={semantic_weight} ---")
            results = similarity.semantic_enhanced_similarity(query, method='hybrid', weights=weights)
            for doc_id, doc_text, score in results[:3]:
                print(f"文档{doc_id} (得分: {score:.4f}): {doc_text[:60]}...")
    
    print("\n\n" + "=" * 80)
    print("                   演示总结")
    print("=" * 80)
    print("\n针对新闻文本中关键词与文本语义相近但不完全匹配的情况，我们建议：")
    print("\n1. 使用深度学习语义相似度方法（如Sentence-BERT）")
    print("   - 这些方法能够捕捉文本的深层语义信息")
    print("   - 对于新闻文本，特别是当关键词与文档内容不完全匹配时，效果显著优于传统方法")
    print("\n2. 使用增强语义相似度方法（混合传统方法和深度学习方法）")
    print("   - 推荐权重设置：BM25=0.3, 语义相似度=0.7")
    print("   - 这种组合既能保留传统方法的精确匹配能力，又能获得语义理解的优势")
    print("\n3. 模型选择建议：")
    print("   - 英文新闻：'all-MiniLM-L6-v2' 或 'all-mpnet-base-v2'")
    print("   - 中文新闻：'uer/sbert-base-chinese-nli' 或 'paraphrase-multilingual-MiniLM-L12-v2'")
    print("\n4. 性能优化建议：")
    print("   - 对于大批量文档，预先计算并缓存嵌入向量")
    print("   - 考虑使用轻量级模型以提高推理速度")
    print("\n\n希望这个演示对您处理新闻文本中的语义相似度问题有所帮助！")
    print("=" * 80)


if __name__ == "__main__":
    news_semantic_similarity_demo()