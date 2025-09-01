import math
import jieba
from collections import Counter, defaultdict
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
# 尝试导入sentence-transformers库，如果没有安装会提供提示


class KeywordTextSimilarity:
    """
    关键词与文本相关性排序算法实现
    包含多种高精度的排序方法：TF-IDF、BM25、词向量相似度
    """
    
    def __init__(self):
        # 初始化停用词表，可以根据需要扩展
        self.stop_words = set(['的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'])
        
        # 用于BM25算法的参数
        self.k1 = 1.2  # 词频饱和参数
        self.b = 0.75  # 文档长度归一化参数
        self.avg_doc_len = 0  # 平均文档长度
        
        # 文档集统计信息
        self.documents = []
        self.doc_terms = []  # 每个文档的词项列表
        self.term_freq = defaultdict(Counter)  # 词项在各文档中的频率
        self.doc_freq = defaultdict(int)  # 词项的文档频率
        self.total_docs = 0  # 总文档数
        
        # 预训练模型相关
        self.bert_model = None
        self.bert_tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 尝试导入sentence-transformers库
        try:
            from sentence_transformers import SentenceTransformer
            self.sentence_transformer_available = True
        except ImportError:
            self.sentence_transformer_available = False
            self.sentence_model = None
    
    def add_document(self, doc_id, document):
        """添加文档到文档集合中"""
        self.documents.append((doc_id, document))
        terms = self._tokenize(document)
        self.doc_terms.append(terms)
        
        # 更新词频统计
        term_counts = Counter(terms)
        for term, count in term_counts.items():
            self.term_freq[doc_id][term] = count
            
        # 更新文档频率
        for term in set(terms):
            self.doc_freq[term] += 1
            
        self.total_docs += 1
    
    def _tokenize(self, text):
        """中文分词并过滤停用词"""
        words = jieba.lcut(text)
        return [word for word in words if word not in self.stop_words and word.strip()]
    
    def _calculate_avg_doc_len(self):
        """计算平均文档长度"""
        if self.total_docs == 0:
            return 0
        total_length = sum(len(terms) for terms in self.doc_terms)
        return total_length / self.total_docs
    
    def tf_idf_similarity(self, query):
        """
        基于TF-IDF算法计算查询词与文档的相关性
        返回按相关性排序的文档列表
        """
        if not self.documents:
            return []
        
        query_terms = self._tokenize(query)
        results = []
        
        for doc_id, document in self.documents:
            doc_terms = self.doc_terms[doc_id]
            doc_score = 0
            
            for term in query_terms:
                # 计算TF
                tf = self.term_freq.get(doc_id, {}).get(term, 0) / len(doc_terms) if doc_terms else 0
                # 计算IDF
                if self.doc_freq.get(term, 0) == 0:
                    idf = 0
                else:
                    idf = math.log(self.total_docs / self.doc_freq[term])
                # 计算TF-IDF分数
                doc_score += tf * idf
            
            results.append((doc_id, document, doc_score))
        
        # 按分数降序排序
        results.sort(key=lambda x: x[2], reverse=True)
        return results
    
    def bm25_similarity(self, query):
        """
        基于BM25算法计算查询词与文档的相关性
        返回按相关性排序的文档列表
        """
        if not self.documents:
            return []
        
        query_terms = self._tokenize(query)
        self.avg_doc_len = self._calculate_avg_doc_len()
        results = []
        
        for doc_id, document in self.documents:
            doc_terms = self.doc_terms[doc_id]
            doc_len = len(doc_terms)
            doc_score = 0
            
            for term in query_terms:
                # 计算IDF
                if self.doc_freq.get(term, 0) == 0:
                    idf = 0
                else:
                    idf = math.log((self.total_docs - self.doc_freq[term] + 0.5) / (self.doc_freq[term] + 0.5) + 1)
                
                # 计算TF项
                tf = self.term_freq.get(doc_id, {}).get(term, 0)
                tf_component = (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len))
                
                # 计算BM25分数
                doc_score += idf * tf_component
            
            results.append((doc_id, document, doc_score))
        
        # 按分数降序排序
        results.sort(key=lambda x: x[2], reverse=True)
        return results
    
    def word2vec_similarity(self, query, embedding_model=None):
        """
        基于词向量计算查询词与文档的相关性
        需要提供预训练的词向量模型
        """
        if not self.documents or embedding_model is None:
            return []
        
        query_terms = self._tokenize(query)
        # 计算查询词向量
        query_vec = self._get_sentence_vector(query_terms, embedding_model)
        if query_vec is None:
            return []
        
        results = []
        for doc_id, document in self.documents:
            doc_terms = self.doc_terms[doc_id]
            doc_vec = self._get_sentence_vector(doc_terms, embedding_model)
            if doc_vec is None:
                results.append((doc_id, document, 0))
                continue
            
            # 计算余弦相似度
            similarity = self._cosine_similarity(query_vec, doc_vec)
            results.append((doc_id, document, similarity))
        
        # 按相似度降序排序
        results.sort(key=lambda x: x[2], reverse=True)
        return results
    
    def _get_sentence_vector(self, terms, embedding_model):
        """计算句子的向量表示"""
        vectors = []
        for term in terms:
            try:
                vectors.append(embedding_model[term])
            except KeyError:
                continue
        
        if not vectors:
            return None
        
        return np.mean(vectors, axis=0)
    
    def _cosine_similarity(self, vec1, vec2):
        """计算两个向量的余弦相似度"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0
        
        return dot_product / (norm1 * norm2)
    
    def hybrid_similarity(self, query, embedding_model=None, weights=(0.4, 0.6)):
        """
        混合BM25和词向量相似度的方法
        weights: (bm25_weight, word2vec_weight)
        """
        if not self.documents:
            return []
        
        # 获取BM25相似度结果
        bm25_results = {doc_id: score for doc_id, _, score in self.bm25_similarity(query)}
        
        # 获取词向量相似度结果（如果提供了模型）
        if embedding_model:
            word2vec_results = {doc_id: score for doc_id, _, score in self.word2vec_similarity(query, embedding_model)}
        else:
            # 如果没有提供词向量模型，只使用BM25
            return self.bm25_similarity(query)
        
        # 归一化分数
        max_bm25 = max(bm25_results.values()) if bm25_results else 1
        max_w2v = max(word2vec_results.values()) if word2vec_results else 1
        
        # 计算混合分数
        hybrid_results = []
        for doc_id, document in self.documents:
            norm_bm25 = bm25_results.get(doc_id, 0) / max_bm25 if max_bm25 else 0
            norm_w2v = word2vec_results.get(doc_id, 0) / max_w2v if max_w2v else 0
            hybrid_score = weights[0] * norm_bm25 + weights[1] * norm_w2v
            hybrid_results.append((doc_id, document, hybrid_score))
        
        # 按混合分数降序排序
        hybrid_results.sort(key=lambda x: x[2], reverse=True)
        return hybrid_results
        
    def load_bert_model(self, model_name='hfl/chinese-roberta-wwm-ext'):
        """
        加载BERT预训练模型
        适用于需要捕获深层语义关系的场景
        """
        try:
            self.bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.bert_model = AutoModel.from_pretrained(model_name).to(self.device)
            self.bert_model.eval()
            print(f"成功加载BERT模型: {model_name}")
            return True
        except Exception as e:
            print(f"加载BERT模型失败: {e}")
            return False
            
    def load_sentence_transformer(self, model_name='all-MiniLM-L6-v2'):
        """
        加载Sentence-BERT预训练模型
        专为句子嵌入和语义相似度计算优化
        """
        if not self.sentence_transformer_available:
            print("sentence-transformers库未安装，请运行: pip install sentence-transformers")
            return False
            
        try:
            from sentence_transformers import SentenceTransformer
            self.sentence_model = SentenceTransformer(model_name, device=self.device)
            print(f"成功加载Sentence-BERT模型: {model_name}")
            return True
        except Exception as e:
            print(f"加载Sentence-BERT模型失败: {e}")
            return False
            
    def _get_bert_embedding(self, text):
        """
        获取文本的BERT嵌入向量
        """
        if self.bert_model is None or self.bert_tokenizer is None:
            print("请先调用load_bert_model方法加载BERT模型")
            return None
            
        inputs = self.bert_tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            # 使用[CLS]标记的嵌入作为句子表示
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
            
        return embedding
            
    def bert_similarity(self, query):
        """
        基于BERT模型计算查询与文档的语义相似度
        适用于新闻文本等需要深层语义理解的场景
        """
        if not self.documents or self.bert_model is None:
            return []
        
        # 计算查询的BERT嵌入
        query_embedding = self._get_bert_embedding(query)
        if query_embedding is None:
            return []
            
        results = []
        for doc_id, document in self.documents:
            # 计算文档的BERT嵌入
            doc_embedding = self._get_bert_embedding(document)
            if doc_embedding is None:
                results.append((doc_id, document, 0))
                continue
                
            # 计算余弦相似度
            similarity = self._cosine_similarity(query_embedding, doc_embedding)
            results.append((doc_id, document, similarity))
            
        # 按相似度降序排序
        results.sort(key=lambda x: x[2], reverse=True)
        return results
            
    def sentence_transformer_similarity(self, query):
        """
        基于Sentence-BERT模型计算查询与文档的语义相似度
        专为句子嵌入和语义相似度计算优化，效果更好
        """
        if not self.documents or not self.sentence_transformer_available or self.sentence_model is None:
            return []
            
        # 计算查询的嵌入
        query_embedding = self.sentence_model.encode(query, convert_to_tensor=True)
        
        results = []
        for doc_id, document in self.documents:
            # 计算文档的嵌入
            doc_embedding = self.sentence_model.encode(document, convert_to_tensor=True)
            
            # 计算余弦相似度
            similarity = self.sentence_model.similarity(query_embedding, doc_embedding).item()
            results.append((doc_id, document, similarity))
            
        # 按相似度降序排序
        results.sort(key=lambda x: x[2], reverse=True)
        return results
            
    def semantic_enhanced_similarity(self, query, method='sentence_transformer', weights=None):
        """
        增强的语义相似度计算方法，专为新闻文本优化
        
        method: 
            'sentence_transformer': 使用Sentence-BERT模型（推荐）
            'bert': 使用标准BERT模型
            'hybrid': 混合传统方法和语义方法
        
        weights: 当method='hybrid'时的权重配置，格式为(bm25_weight, semantic_weight)
        """
        if not self.documents:
            return []
            
        # 使用Sentence-BERT模型（推荐方法）
        if method == 'sentence_transformer':
            if self.sentence_model:
                return self.sentence_transformer_similarity(query)
            else:
                print("Sentence-BERT模型未加载，请先调用load_sentence_transformer方法")
                # 如果Sentence-BERT不可用，尝试使用BERT
                return self.bert_similarity(query) if self.bert_model else []
                
        # 使用标准BERT模型
        elif method == 'bert':
            return self.bert_similarity(query)
            
        # 使用混合方法
        elif method == 'hybrid':
            if weights is None:
                weights = (0.3, 0.7)  # 默认给语义相似度更高的权重
                
            # 获取BM25结果
            bm25_results = {doc_id: score for doc_id, _, score in self.bm25_similarity(query)}
            
            # 获取语义相似度结果（优先使用Sentence-BERT）
            if self.sentence_model:
                semantic_results = {doc_id: score for doc_id, _, score in self.sentence_transformer_similarity(query)}
            elif self.bert_model:
                semantic_results = {doc_id: score for doc_id, _, score in self.bert_similarity(query)}
            else:
                print("没有可用的语义模型，返回BM25结果")
                return self.bm25_similarity(query)
                
            # 归一化分数
            max_bm25 = max(bm25_results.values()) if bm25_results else 1
            max_semantic = max(semantic_results.values()) if semantic_results else 1
            
            # 计算混合分数
            hybrid_results = []
            for doc_id, document in self.documents:
                norm_bm25 = bm25_results.get(doc_id, 0) / max_bm25 if max_bm25 else 0
                norm_semantic = semantic_results.get(doc_id, 0) / max_semantic if max_semantic else 0
                hybrid_score = weights[0] * norm_bm25 + weights[1] * norm_semantic
                hybrid_results.append((doc_id, document, hybrid_score))
                
            # 按混合分数降序排序
            hybrid_results.sort(key=lambda x: x[2], reverse=True)
            return hybrid_results
            
        else:
            print("不支持的方法，请选择'sentence_transformer'、'bert'或'hybrid'")
            return []

# 示例用法
def demo():
    # 创建相似度计算实例
    similarity = KeywordTextSimilarity()
    
    # 添加一些示例文档（新闻文本示例）
    documents = [
        (0, "国家航天局宣布，我国首次火星探测任务天问一号探测器成功着陆火星乌托邦平原南部预选着陆区，迈出了我国星际探测征程的重要一步。"),
        (1, "国务院总理主持召开国务院常务会议，部署进一步支持小微企业、个体工商户纾困和发展的措施，确保市场主体稳定运行。"),
        (2, "世界卫生组织发布最新疫情报告，全球累计确诊病例超过1.5亿例，各国正加速推进疫苗接种工作以控制疫情传播。"),
        (3, "国际能源署发布报告显示，随着各国加大可再生能源投资，全球清洁能源转型步伐正在加快，有望在未来十年显著减少碳排放。"),
        (4, "我国科研人员在量子计算领域取得重大突破，成功构建76个光子的量子计算原型机九章，计算能力较传统超级计算机提升百亿倍。")
    ]
    
    for doc_id, doc_text in documents:
        similarity.add_document(doc_id, doc_text)
    
    # 查询关键词（可能与文档不完全匹配但语义相关）
    query = "航天探索 太空任务"
    
    print("=== 查询关键词: {}".format(query))
    print("\n--- 传统方法结果 (BM25) ---")
    # 使用传统BM25算法
    bm25_results = similarity.bm25_similarity(query)
    for doc_id, doc_text, score in bm25_results[:3]:
        print(f"文档{doc_id} (得分: {score:.4f}): {doc_text}")
    
    print("\n--- 提示：为了处理语义相近但不完全匹配的情况，我们可以使用深度学习模型 ---")
    print("\n1. 您可以使用BERT模型：")
    print("   similarity.load_bert_model()")
    print("   bert_results = similarity.bert_similarity(query)")
    
    print("\n2. 推荐使用Sentence-BERT模型（需要安装sentence-transformers）：")
    print("   similarity.load_sentence_transformer()")
    print("   st_results = similarity.sentence_transformer_similarity(query)")
    
    print("\n3. 使用增强的语义相似度方法（专为新闻文本优化）：")
    print("   # 推荐方法：sentence_transformer")
    print("   results = similarity.semantic_enhanced_similarity(query, method='sentence_transformer')")
    print("   # 或使用混合方法")
    print("   results = similarity.semantic_enhanced_similarity(query, method='hybrid', weights=(0.3, 0.7))")
    
    print("\n--- 安装依赖提示 ---")
    print("如需使用深度学习模型，请安装额外依赖：")
    print("pip install transformers torch sentence-transformers")

if __name__ == "__main__":
    demo()