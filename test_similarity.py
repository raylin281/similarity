from keyword_similarity import KeywordTextSimilarity
import time

def test_keyword_similarity():
    """
    测试关键词与文本相关性排序算法
    展示不同算法的使用方法和效果，特别是针对新闻文本中关键词与文本语义相近但不完全匹配的情况
    """
    # 创建相似度计算实例
    similarity = KeywordTextSimilarity()
    
    # 准备测试文档集 - 新闻文本示例
    documents = [
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
    for doc_id, doc_text in documents:
        similarity.add_document(doc_id, doc_text)
    print(f"已添加{len(documents)}个文档。\n")
    
    # 定义多个测试查询 - 包括精确匹配和语义相近的情况
    queries = [
        "火星探测 航天任务",  # 与文档0语义相关但不完全匹配
        "小微企业 支持政策",   # 与文档1语义相关但不完全匹配
        "气候变化 健康影响",   # 与文档2语义相关但不完全匹配
        "清洁能源 能源转型",   # 与文档3语义相关但不完全匹配
        "量子计算 科技突破"    # 与文档6语义相关但不完全匹配
    ]
    
    # 对每个查询进行测试
    for query in queries:
        print(f"\n\n========== 查询关键词: '{query}' ==========")
        
        # 使用TF-IDF算法
        print("\n--- TF-IDF 结果 ---")
        tf_idf_results = similarity.tf_idf_similarity(query)
        for doc_id, doc_text, score in tf_idf_results[:3]:
            print(f"文档{doc_id} (得分: {score:.4f}): {doc_text[:50]}...")
        
        # 使用BM25算法
        print("\n--- BM25 结果 ---")
        bm25_results = similarity.bm25_similarity(query)
        for doc_id, doc_text, score in bm25_results[:3]:
            print(f"文档{doc_id} (得分: {score:.4f}): {doc_text[:50]}...")
        
        # 测试深度学习语义相似度方法
        try:
            # 加载Sentence-BERT模型
            print("\n--- 正在加载Sentence-BERT模型... ---")
            start_time = time.time()
            similarity.load_sentence_transformer()
            load_time = time.time() - start_time
            print(f"模型加载完成，耗时: {load_time:.2f}秒")
            
            # 使用Sentence-BERT语义相似度
            print("\n--- Sentence-BERT 语义相似度结果 ---")
            sentence_bert_results = similarity.sentence_transformer_similarity(query)
            for doc_id, doc_text, score in sentence_bert_results[:3]:
                print(f"文档{doc_id} (得分: {score:.4f}): {doc_text[:50]}...")
            
            # 使用增强语义相似度方法（混合BM25和Sentence-BERT）
            print("\n--- 增强语义相似度 (混合BM25 + Sentence-BERT) 结果 ---")
            enhanced_results = similarity.semantic_enhanced_similarity(query, method='hybrid', weights=(0.3, 0.7))
            for doc_id, doc_text, score in enhanced_results[:3]:
                print(f"文档{doc_id} (得分: {score:.4f}): {doc_text[:50]}...")
                
        except Exception as e:
            print(f"\n--- 深度学习模型测试失败，错误信息: {str(e)} ---")
            print("请确保已安装所需依赖: pip install transformers torch sentence-transformers")
    
    print("\n\n测试完成！")
    print("\n注意：")
    print("1. 对于新闻文本处理，特别是关键词与文本不完全匹配但语义相关的情况，推荐使用Sentence-BERT或增强语义相似度方法")
    print("2. 如需使用词向量方法，请加载预训练的词向量模型")
    print("3. 对于中文新闻文本，可以尝试使用中文优化的模型：uer/sbert-base-chinese-nli")

if __name__ == "__main__":
    test_keyword_similarity()