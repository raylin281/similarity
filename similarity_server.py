import json
import logging
from flask import Flask, request, jsonify
from keyword_similarity import KeywordTextSimilarity

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 全局相似度模型实例
similarity_model = None
model_initialized = False

@app.before_request
def init_model():
    """在首次请求前初始化相似度模型"""
    global similarity_model, model_initialized
    if not model_initialized:
        similarity_model = KeywordTextSimilarity()
        model_initialized = True
        logger.info("相似度模型初始化完成")

@app.route('/api/load_documents', methods=['POST'])
def load_documents():
    """
    加载文档到模型中
    请求格式: {
        "documents": [
            {"id": 0, "text": "文档文本内容"},
            ...
        ]
    }
    """
    try:
        data = request.json
        documents = data.get('documents', [])
        
        if not documents:
            return jsonify({"success": False, "message": "文档列表不能为空"}), 400
        
        # 清空现有文档并添加新文档
        global similarity_model
        similarity_model = KeywordTextSimilarity()  # 重新初始化以清空之前的文档
        
        for doc in documents:
            doc_id = doc.get('id')
            doc_text = doc.get('text')
            if doc_id is not None and doc_text:
                similarity_model.add_document(doc_id, doc_text)
        
        logger.info(f"成功加载 {len(documents)} 个文档")
        return jsonify({"success": True, "message": f"成功加载 {len(documents)} 个文档"})
    except Exception as e:
        logger.error(f"加载文档失败: {str(e)}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/load_model', methods=['POST'])
def load_model():
    """
    加载深度学习模型
    请求格式: {
        "model_type": "sentence_transformer",  # 可选值: "sentence_transformer", "bert"
        "model_name": "all-MiniLM-L6-v2"       # 模型名称
    }
    """
    try:
        data = request.json
        model_type = data.get('model_type', 'sentence_transformer')
        model_name = data.get('model_name', 'all-MiniLM-L6-v2')
        
        global similarity_model
        if model_type == 'sentence_transformer':
            result = similarity_model.load_sentence_transformer(model_name)
            if result:
                logger.info(f"成功加载Sentence-BERT模型: {model_name}")
                return jsonify({"success": True, "message": f"成功加载Sentence-BERT模型: {model_name}"})
            else:
                return jsonify({"success": False, "message": "加载Sentence-BERT模型失败，请检查是否安装了sentence-transformers库"}), 400
        elif model_type == 'bert':
            result = similarity_model.load_bert_model(model_name)
            if result:
                logger.info(f"成功加载BERT模型: {model_name}")
                return jsonify({"success": True, "message": f"成功加载BERT模型: {model_name}"})
            else:
                return jsonify({"success": False, "message": "加载BERT模型失败"}), 400
        else:
            return jsonify({"success": False, "message": "不支持的模型类型，可选值: sentence_transformer, bert"}), 400
    except Exception as e:
        logger.error(f"加载模型失败: {str(e)}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/calculate_similarity', methods=['POST'])
def calculate_similarity():
    """
    计算查询与文档的相似度
    请求格式: {
        "query": "查询关键词",
        "method": "hybrid",  # 可选值: "bm25", "bert", "sentence_transformer", "hybrid"
        "top_k": 10,         # 返回的最相关文档数量
        "weights": [0.3, 0.7] # 当method为hybrid时的权重配置
    }
    """
    try:
        data = request.json
        query = data.get('query', '')
        method = data.get('method', 'hybrid')
        top_k = data.get('top_k', 10)
        weights = data.get('weights', [0.3, 0.7])
        
        if not query:
            return jsonify({"success": False, "message": "查询关键词不能为空"}), 400
        
        global similarity_model
        
        # 根据选择的方法计算相似度
        if method == 'bm25':
            results = similarity_model.bm25_similarity(query)
        elif method == 'bert':
            results = similarity_model.bert_similarity(query)
        elif method == 'sentence_transformer':
            results = similarity_model.sentence_transformer_similarity(query)
        elif method == 'hybrid':
            results = similarity_model.semantic_enhanced_similarity(query, method='hybrid', weights=tuple(weights))
        else:
            return jsonify({"success": False, "message": "不支持的相似度计算方法"}), 400
        
        # 格式化结果
        formatted_results = []
        for doc_id, doc_text, score in results[:top_k]:
            formatted_results.append({
                "id": doc_id,
                "text": doc_text,
                "score": round(float(score), 6)
            })
        
        return jsonify({"success": True, "results": formatted_results})
    except Exception as e:
        logger.error(f"计算相似度失败: {str(e)}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({"success": True, "status": "running"})

if __name__ == '__main__':
    # 在生产环境中应该使用WSGI服务器如Gunicorn
    # 这里仅作为开发测试使用
    app.run(host='0.0.0.0', port=5000, debug=False)