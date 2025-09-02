# 关键词相似度模型API服务说明

本项目提供了一个基于HTTP的REST API服务，将`KeywordTextSimilarity`模型封装为可被Java后端调用的接口。

## 目录结构

```
/root/autodl-fs/similarity/
├── keyword_similarity.py      # 核心相似度计算模型
├── similarity_server.py       # Flask API服务封装
├── requirements.txt           # 项目依赖
└── README_API.md              # API使用说明
```

## 部署步骤

1. **安装依赖**

```bash
pip install -r requirements.txt
```

2. **启动API服务**

```bash
python similarity_server.py
```

服务将在默认端口`5000`上启动，监听所有IP地址(`0.0.0.0`)。

在生产环境中，建议使用WSGI服务器如Gunicorn来部署：

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 similarity_server:app
```

## API接口说明

### 1. 健康检查接口

**请求URL**：`/api/health`

**请求方法**：GET

**返回示例**：

```json
{
  "success": true,
  "status": "running"
}
```

### 2. 加载文档接口

**请求URL**：`/api/load_documents`

**请求方法**：POST

**请求体**：

```json
{
  "documents": [
    {"id": 0, "text": "文档文本内容"},
    {"id": 1, "text": "另一篇文档内容"}
  ]
}
```

**返回示例**：

```json
{
  "success": true,
  "message": "成功加载 2 个文档"
}
```

### 3. 加载深度学习模型接口

**请求URL**：`/api/load_model`

**请求方法**：POST

**请求体**：

```json
{
  "model_type": "sentence_transformer",  # 可选值: "sentence_transformer", "bert"
  "model_name": "all-MiniLM-L6-v2"       # 模型名称
}
```

**返回示例**：

```json
{
  "success": true,
  "message": "成功加载Sentence-BERT模型: all-MiniLM-L6-v2"
}
```

### 4. 计算相似度接口

**请求URL**：`/api/calculate_similarity`

**请求方法**：POST

**请求体**：

```json
{
  "query": "火星探测 航天任务",
  "method": "hybrid",  # 可选值: "bm25", "bert", "sentence_transformer", "hybrid"
  "top_k": 10,         # 返回的最相关文档数量
  "weights": [0.3, 0.7] # 当method为hybrid时的权重配置
}
```

**返回示例**：

```json
{
  "success": true,
  "results": [
    {
      "id": 0,
      "text": "国家航天局宣布，我国首次火星探测任务天问一号探测器成功着陆火星乌托邦平原南部预选着陆区...",
      "score": 0.926743
    },
    {
      "id": 4,
      "text": "中国科学院宣布，在量子计算领域取得重大突破，成功构建76个光子的量子计算原型机...",
      "score": 0.451289
    }
  ]
}
```

## Java后端调用示例

下面是Java代码示例，展示如何调用我们部署的相似度计算服务：

```java
import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONArray;
import com.alibaba.fastjson.JSONObject;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

public class SimilarityServiceClient {
    
    // API服务基础URL
    private static final String BASE_URL = "http://localhost:5000/api";
    
    /**
     * 发送HTTP POST请求
     */
    private static String sendPostRequest(String endpoint, String jsonBody) throws Exception {
        URL url = new URL(BASE_URL + endpoint);
        HttpURLConnection connection = (HttpURLConnection) url.openConnection();
        
        // 设置请求属性
        connection.setRequestMethod("POST");
        connection.setRequestProperty("Content-Type", "application/json; utf-8");
        connection.setRequestProperty("Accept", "application/json");
        connection.setDoOutput(true);
        
        // 发送请求体
        try (OutputStream os = connection.getOutputStream()) {
            byte[] input = jsonBody.getBytes(StandardCharsets.UTF_8);
            os.write(input, 0, input.length);
        }
        
        // 读取响应
        try (BufferedReader br = new BufferedReader(
                new InputStreamReader(connection.getInputStream(), StandardCharsets.UTF_8))) {
            StringBuilder response = new StringBuilder();
            String responseLine;
            while ((responseLine = br.readLine()) != null) {
                response.append(responseLine.trim());
            }
            return response.toString();
        } finally {
            connection.disconnect();
        }
    }
    
    /**
     * 加载文档到相似度模型
     */
    public static boolean loadDocuments(List<Document> documents) throws Exception {
        JSONObject requestBody = new JSONObject();
        JSONArray docsArray = new JSONArray();
        
        for (Document doc : documents) {
            JSONObject docObj = new JSONObject();
            docObj.put("id", doc.getId());
            docObj.put("text", doc.getText());
            docsArray.add(docObj);
        }
        
        requestBody.put("documents", docsArray);
        String response = sendPostRequest("/load_documents", requestBody.toJSONString());
        
        JSONObject jsonResponse = JSON.parseObject(response);
        return jsonResponse.getBoolean("success");
    }
    
    /**
     * 加载深度学习模型
     */
    public static boolean loadModel(String modelType, String modelName) throws Exception {
        JSONObject requestBody = new JSONObject();
        requestBody.put("model_type", modelType);
        requestBody.put("model_name", modelName);
        
        String response = sendPostRequest("/load_model", requestBody.toJSONString());
        JSONObject jsonResponse = JSON.parseObject(response);
        return jsonResponse.getBoolean("success");
    }
    
    /**
     * 计算查询与文档的相似度
     */
    public static List<SimilarityResult> calculateSimilarity(String query, String method, int topK) throws Exception {
        JSONObject requestBody = new JSONObject();
        requestBody.put("query", query);
        requestBody.put("method", method);
        requestBody.put("top_k", topK);
        
        // 如果使用混合方法，可以设置权重
        if ("hybrid".equals(method)) {
            JSONArray weights = new JSONArray();
            weights.add(0.3);
            weights.add(0.7);
            requestBody.put("weights", weights);
        }
        
        String response = sendPostRequest("/calculate_similarity", requestBody.toJSONString());
        JSONObject jsonResponse = JSON.parseObject(response);
        
        if (!jsonResponse.getBoolean("success")) {
            throw new Exception("计算相似度失败: " + jsonResponse.getString("message"));
        }
        
        // 解析结果
        List<SimilarityResult> results = new ArrayList<>();
        JSONArray resultsArray = jsonResponse.getJSONArray("results");
        
        for (Object obj : resultsArray) {
            JSONObject resultObj = (JSONObject) obj;
            SimilarityResult result = new SimilarityResult();
            result.setId(resultObj.getInteger("id"));
            result.setText(resultObj.getString("text"));
            result.setScore(resultObj.getDouble("score"));
            results.add(result);
        }
        
        return results;
    }
    
    // 文档类定义
    public static class Document {
        private int id;
        private String text;
        
        public Document(int id, String text) {
            this.id = id;
            this.text = text;
        }
        
        public int getId() { return id; }
        public void setId(int id) { this.id = id; }
        public String getText() { return text; }
        public void setText(String text) { this.text = text; }
    }
    
    // 相似度结果类定义
    public static class SimilarityResult {
        private int id;
        private String text;
        private double score;
        
        public int getId() { return id; }
        public void setId(int id) { this.id = id; }
        public String getText() { return text; }
        public void setText(String text) { this.text = text; }
        public double getScore() { return score; }
        public void setScore(double score) { this.score = score; }
    }
    
    // 示例使用方法
    public static void main(String[] args) {
        try {
            // 1. 准备文档
            List<Document> documents = new ArrayList<>();
            documents.add(new Document(0, "国家航天局宣布，我国首次火星探测任务天问一号探测器成功着陆火星乌托邦平原南部预选着陆区，迈出了我国星际探测征程的重要一步。"));
            documents.add(new Document(1, "国务院总理主持召开国务院常务会议，部署进一步支持小微企业、个体工商户纾困和发展的措施。"));
            
            // 2. 加载文档
            boolean docsLoaded = loadDocuments(documents);
            System.out.println("文档加载结果: " + docsLoaded);
            
            // 3. 加载深度学习模型
            boolean modelLoaded = loadModel("sentence_transformer", "all-MiniLM-L6-v2");
            System.out.println("模型加载结果: " + modelLoaded);
            
            // 4. 计算相似度
            List<SimilarityResult> results = calculateSimilarity("火星探测 航天任务", "hybrid", 10);
            
            // 5. 处理结果
            System.out.println("相似度计算结果:");
            for (SimilarityResult result : results) {
                System.out.println("文档ID: " + result.getId() + ", 得分: " + result.getScore());
                System.out.println("文本: " + result.getText());
                System.out.println("-------------------");
            }
            
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
```

## 注意事项

1. **模型加载**：深度学习模型（Sentence-BERT或BERT）需要在首次使用前加载，这可能需要一些时间和网络连接（用于下载预训练模型）。

2. **性能优化**：
   - 对于大量文档的场景，建议在服务启动时预先加载文档。
   - 在生产环境中，推荐使用Gunicorn等WSGI服务器，并配置适当数量的工作进程。
   - 考虑添加缓存机制，避免重复计算相同查询的相似度。

3. **错误处理**：API接口返回统一的错误格式，包含`success`标志和`message`字段，请确保Java客户端正确处理这些错误信息。

4. **模型选择**：
   - 对于中文文本，建议使用中文优化的预训练模型，如`uer/sbert-base-chinese-nli`。
   - 在精度和性能之间需要权衡，`sentence_transformer`方法通常提供最佳的语义匹配效果。

## 扩展建议

1. 添加API认证机制，如JWT token验证。
2. 实现批量处理功能，提高处理效率。
3. 添加日志记录和监控，便于问题排查和性能优化。
4. 考虑使用Docker容器化部署，简化环境配置。