"""
学术文献检索 + 文档问答 Web 应用

"""

import os
import sys
import re
import httpx
from datetime import datetime
from dotenv import load_dotenv

import streamlit as st

# 加载本地 .env
load_dotenv()


def get_resource_path(relative_path):
  
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)


# ============ 页面配置 ============
st.set_page_config(
    page_title="学术文献检索助手",
    page_icon="📚",
    layout="wide",
)

# ============ API Key 管理 ============
def get_deepseek_key():
    if "DEEPSEEK_API_KEY" in st.secrets:
        return st.secrets["DEEPSEEK_API_KEY"]
    if os.getenv("LLM_API_KEY"):
        return os.getenv("LLM_API_KEY")
    return st.session_state.get("deepseek_key", "")

def get_elsevier_key():
    if "ELSEVIER_API_KEY" in st.secrets:
        return st.secrets["ELSEVIER_API_KEY"]
    if os.getenv("SCIENCEDIRECT_API_KEY"):
        return os.getenv("SCIENCEDIRECT_API_KEY")
    return st.session_state.get("elsevier_key", "")

# ============ Scopus API 函数 ============
SEARCH_URL = "https://api.elsevier.com/content/search/scopus"

def clean_keyword(keyword):
    keyword = keyword.replace('"', "").replace("'", "")
    keyword = re.sub(r"\s+(AND|OR|NOT)\s+", " ", keyword, flags=re.IGNORECASE)
    return " ".join(keyword.split())

def format_authors(author_str, max_show=3):
    if not author_str:
        return "未知"
    authors = [a.strip() for a in author_str.split(",") if a.strip()]
    result = ", ".join(authors[:max_show])
    if len(authors) > max_show:
        result += f" 等{len(authors)}人"
    return result

def search_papers(keyword, limit=10, year_from=None, year_to=None):
    """搜索论文"""
    api_key = get_elsevier_key()
    if not api_key:
        return "❌ 未配置 Elsevier API Key"
    
    kw = clean_keyword(keyword)
    query = f"TITLE-ABS-KEY({kw})"
    if year_from:
        query += f" AND PUBYEAR > {year_from - 1}"
    if year_to:
        query += f" AND PUBYEAR < {year_to + 1}"
    
    params = {
        "query": query,
        "count": min(limit, 25),
        "sort": "-relevancy",
        "field": "dc:title,dc:creator,prism:publicationName,prism:coverDate,prism:doi,citedby-count,dc:description"
    }
    headers = {"X-ELS-APIKey": api_key, "Accept": "application/json"}
    
    try:
        resp = httpx.get(SEARCH_URL, params=params, headers=headers, timeout=30)
        if resp.status_code != 200:
            return f"❌ API 返回 {resp.status_code}"
        
        data = resp.json()
        entries = data.get("search-results", {}).get("entry", [])
        total = data.get("search-results", {}).get("opensearch:totalResults", "0")
        entries = [e for e in entries if e.get("dc:title")]
        
        if not entries:
            return f"没有找到与「{kw}」相关的论文。"
        
        result = {"total": total, "papers": []}
        for e in entries:
            result["papers"].append({
                "title": e.get("dc:title", ""),
                "authors": format_authors(e.get("dc:creator", "")),
                "journal": e.get("prism:publicationName", ""),
                "year": e.get("prism:coverDate", "")[:4],
                "doi": e.get("prism:doi", ""),
                "cited": e.get("citedby-count", "0"),
                "abstract": e.get("dc:description", ""),
            })
        return result
    except Exception as e:
        return f"❌ 搜索出错：{e}"

def get_abstract_by_doi(doi):
    """通过 DOI 获取完整摘要（"""
    api_key = get_elsevier_key()
    if not api_key:
        return "❌ 未配置 Elsevier API Key"
    
    params = {
        "query": f"DOI({doi})",
        "count": 1,
        "field": "dc:title,dc:creator,prism:publicationName,prism:coverDate,prism:doi,citedby-count,dc:description"
    }
    headers = {"X-ELS-APIKey": api_key, "Accept": "application/json"}
    
    try:
        resp = httpx.get(SEARCH_URL, params=params, headers=headers, timeout=30)
        if resp.status_code != 200:
            return f"❌ 查询失败（HTTP {resp.status_code}）"
        
        entries = resp.json().get("search-results", {}).get("entry", [])
        if not entries:
            return f"❌ 找不到 DOI 为 {doi} 的论文。"
        
        e = entries[0]
        abstract = e.get("dc:description", "")
        if not abstract:
            abstract = "（该论文暂无摘要）"
        
        return {
            "title": e.get("dc:title", ""),
            "authors": format_authors(e.get("dc:creator", ""), 20),
            "journal": e.get("prism:publicationName", ""),
            "year": e.get("prism:coverDate", "")[:4],
            "cited": e.get("citedby-count", "0"),
            "doi": e.get("prism:doi", doi),
            "abstract": abstract,
        }
    except Exception as e:
        return f"❌ 获取出错：{e}"

# ============ RAG 文档问答引擎 ============
@st.cache_resource(show_spinner=False)
def get_rag_engine():
    """初始化 RAG 引擎（只初始化一次）"""
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
    from langchain_openai import ChatOpenAI
    import tempfile
    
    class RAGEngine:
        def __init__(self):
       
            local_model_path = get_resource_path("models/all-MiniLM-L6-v2")
            if os.path.exists(local_model_path):
                model_name = local_model_path
            else:
                model_name = "sentence-transformers/all-MiniLM-L6-v2"
            
            self.embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            self.llm = ChatOpenAI(
                model=os.getenv("LLM_MODEL", "deepseek-chat"),
                api_key=get_deepseek_key(),
                base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com"),
                temperature=0,
            )
            self.vector_store = None
            self.loaded_files = []
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500, chunk_overlap=100,
                separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""],
            )
        
        def load_document(self, uploaded_file):
            """加载上传的文档"""
            import tempfile
            suffix = os.path.splitext(uploaded_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            
            try:
                if suffix == ".pdf":
                    loader = PyPDFLoader(tmp_path)
                    docs = loader.load()
                elif suffix == ".txt":
                    loader = TextLoader(tmp_path, encoding="utf-8")
                    docs = loader.load()
                elif suffix in [".docx", ".doc"]:
                    loader = Docx2txtLoader(tmp_path)
                    docs = loader.load()
                else:
                    return f"❌ 不支持的格式：{suffix}"
                
                splits = self.text_splitter.split_documents(docs)
                
                if self.vector_store is None:
                    self.vector_store = FAISS.from_documents(splits, self.embeddings)
                else:
                    self.vector_store.add_documents(splits)
                
                self.loaded_files.append(uploaded_file.name)
                return f"✅ 已加载「{uploaded_file.name}」，共 {len(splits)} 个文本块"
            except Exception as e:
                return f"❌ 加载失败：{e}"
            finally:
                os.unlink(tmp_path)
        
        def query(self, question, top_k=4):
            """基于文档回答问题"""
            if self.vector_store is None:
                return "❌ 还没有加载任何文档。"
            
            docs = self.vector_store.similarity_search(question, k=top_k)
            if not docs:
                return "没有找到相关内容。"
            
            context = "\n\n".join([
                f"[片段{i+1}]\n{doc.page_content}"
                for i, doc in enumerate(docs)
            ])
            
            prompt = f"""请根据以下文档内容回答用户的问题。
要求：
1. 只基于提供的文档内容回答，不要编造文档中没有的信息
2. 如果文档中没有相关内容，直接说"文档中没有相关信息"
3. 回答要简洁准确，引用文档原文时用引号标注
4. 每个结论后面标注来自哪个片段，如[片段1]

已加载文档：{', '.join(self.loaded_files)}

---
文档内容：
{context}
---

用户问题：{question}

回答："""
            response = self.llm.invoke(prompt)
            return response.content
        
        def list_files(self):
            if not self.loaded_files:
                return "当前没有加载任何文档。"
            return f"已加载 {len(self.loaded_files)} 个文档：\n" + "\n".join(
                f"  {i+1}. {f}" for i, f in enumerate(self.loaded_files)
            )
        
        def clear(self):
            self.vector_store = None
            self.loaded_files = []
            return "✅ 已清空所有文档。"
    
    return RAGEngine()

# ============ AI 文献检索 ============
def ai_search(question):
    """AI 自动拆解问题，多关键词搜索，获取完整摘要，汇总结果"""
    from langchain_openai import ChatOpenAI
    from langchain_core.tools import tool
    from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
    
    deepseek_key = get_deepseek_key()
    if not deepseek_key:
        return None, "❌ 未配置 DeepSeek API Key"
    
    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL", "deepseek-chat"),
        api_key=deepseek_key,
        base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com"),
        temperature=0,
    )
    
    @tool
    def tool_search(keyword: str, limit: int = 8) -> str:
        """搜索学术论文，传入英文关键词。返回论文标题、作者、期刊、年份、DOI和摘要片段。"""
        result = search_papers(keyword, limit=limit)
        if isinstance(result, str):
            return result
        text = f"找到约 {result['total']} 篇：\n\n"
        for i, p in enumerate(result["papers"], 1):
            text += f"【{i}】{p['title']}\n"
            text += f"    作者：{p['authors']}\n"
            text += f"    期刊：{p['journal']} | {p['year']} | 引用 {p['cited']}\n"
            if p['doi']:
                text += f"    DOI: {p['doi']}\n"
            if p['abstract']:
                text += f"    摘要片段：{p['abstract'][:500]}\n"
            else:
                text += f"    摘要片段：（无）\n"
            text += "\n"
        return text
    
    @tool
    def tool_get_abstract(doi: str) -> str:
        """根据 DOI 获取论文完整摘要。必须对最相关的3-5篇论文调用此工具获取完整摘要后再回答。"""
        result = get_abstract_by_doi(doi)
        if isinstance(result, str):
            return result
        return f"""标题：{result['title']}
作者：{result['authors']}
期刊：{result['journal']} | {result['year']} | 引用 {result['cited']}
DOI：{result['doi']}
完整摘要：
{result['abstract']}"""
    
    tools = [tool_search, tool_get_abstract]
    llm_with_tools = llm.bind_tools(tools)
    
    system_prompt = """你是学术文献检索助手。

当用户提出研究问题时，严格按以下步骤执行：

1. 拆解问题，提取3-5个关键概念，翻译成英文
2. 生成3-5组不同角度的英文关键词
3. 用 tool_search 分别搜索（每次一个关键词）
4. 【必须】从搜索结果中挑选最相关的3-5篇论文，用 tool_get_abstract 获取它们的完整摘要
   - 即使搜索结果里有摘要片段，也必须调用 tool_get_abstract 获取完整摘要
   - 至少获取3篇论文的完整摘要
5. 基于论文的完整摘要内容 + 标题信息 + 你的专业知识，综合回答用户问题

【回答格式要求】
- 正文中每个结论后面用 [1] [2] 标注引用的文献编号
- 例如："钒溶液除铁主要采用沉淀法和萃取法[1][2]"
- 如果结论来自多篇论文，用 [1][2][3] 标注
- 如果是你自己的知识推断，标注"（基于原理推断）"
- 回答要分点论述，逻辑清晰

【最后必须列出完整参考文献】
格式如下：
## 参考文献
[1] 作者. 标题. 期刊, 年份. DOI: xxx
[2] 作者. 标题. 期刊, 年份. DOI: xxx
...

每篇文献必须包含：作者、标题、期刊、年份、DOI（如果有）

关键词用简单英文词组，不要加 AND/OR/引号。

重要：不获取完整摘要就回答是不合格的，必须先调用 tool_get_abstract。"""
    
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=question)]
    
    response = llm_with_tools.invoke(messages)
    messages.append(response)
    
    max_rounds = 8
    for _ in range(max_rounds):
        if not response.tool_calls:
            break
        for tc in response.tool_calls:
            tool = next((t for t in tools if t.name == tc["name"]), None)
            if tool:
                output = tool.invoke(tc["args"])
            else:
                output = "找不到工具"
            messages.append(ToolMessage(content=str(output), tool_call_id=tc["id"]))
        response = llm_with_tools.invoke(messages)
        messages.append(response)
    
    return response.content, None

# ============ 主界面 ============
def main():
    st.title("📚 学术文献检索 + 文档问答助手")
    st.markdown("---")
    
    with st.sidebar:
        st.header("⚙️ 配置")
        
        deepseek_key = get_deepseek_key()
        elsevier_key = get_elsevier_key()
        
        if not deepseek_key:
            st.text_input("DeepSeek API Key", type="password", key="deepseek_key",
                         placeholder="sk-...")
        else:
            st.success("✅ DeepSeek Key 已配置")
        
        if not elsevier_key:
            st.text_input("Elsevier API Key", type="password", key="elsevier_key",
                         placeholder="在 https://dev.elsevier.com/ 免费申请")
        else:
            st.success("✅ Elsevier Key 已配置")
        
        st.markdown("---")
        st.caption("没有 Key？去申请：")
        st.caption("- DeepSeek: platform.deepseek.com")
        st.caption("- Elsevier: dev.elsevier.com")
    
    if not get_deepseek_key() or not get_elsevier_key():
        st.warning("⚠️ 请在左侧边栏配置 API Key 后使用")
        return
    
    tab1, tab2 = st.tabs(["🔍 文献检索", "📄 文档问答"])
    
    with tab1:
        st.subheader("智能文献检索", anchor=False)
        st.write("用中文提问，AI 会自动拆解关键词、多角度搜索、获取完整摘要、带引用标注汇总结果")
        
        question = st.text_area("输入你的研究问题", height=80,
                               placeholder="例如：为什么碳酸氢钠浸出液除硅效果比钒渣钠化焙烧浸出液差？")
        
        col1, col2 = st.columns([1, 5])
        with col1:
            search_btn = st.button("🔍 开始检索", type="primary", use_container_width=True)
        
        if search_btn and question.strip():
            with st.spinner("AI 正在拆解问题、检索文献并获取完整摘要...（约30-60秒）"):
                answer, error = ai_search(question.strip())
                if error:
                    st.error(error)
                else:
                    st.markdown("### 📋 检索结果")
                    st.markdown(answer)
    
    with tab2:
        st.subheader("文档问答（RAG）", anchor=False)
        st.write("上传 PDF/TXT/DOCX 文档，基于文档内容回答问题")
        
        with st.spinner("正在加载嵌入模型（首次运行需要下载，请稍候）..."):
            try:
                rag = get_rag_engine()
            except Exception as e:
                st.error(f"嵌入模型加载失败：{e}")
                st.info("正在从 HuggingFace 下载嵌入模型，首次加载需要 1-2 分钟，请刷新页面重试...")
                return
        
        uploaded_files = st.file_uploader("上传文档（支持 PDF、TXT、DOCX，可多选）",
                                         type=["pdf", "txt", "docx"],
                                         accept_multiple_files=True)
        
        if uploaded_files:
            for f in uploaded_files:
                if f.name not in rag.loaded_files:
                    with st.spinner(f"正在加载 {f.name}..."):
                        result = rag.load_document(f)
                        if "✅" in result:
                            st.success(result)
                        else:
                            st.error(result)
        
        if rag.loaded_files:
            st.info(rag.list_files())
            col1, col2 = st.columns([1, 5])
            with col1:
                if st.button("🗑️ 清空文档"):
                    rag.clear()
                    st.rerun()
            
            st.markdown("---")
            doc_question = st.text_input("基于文档提问",
                                        placeholder="例如：这篇论文的最佳实验条件是什么？")
            if doc_question:
                with st.spinner("正在检索文档并生成回答..."):
                    answer = rag.query(doc_question)
                    st.markdown("### 🤖 回答")
                    st.markdown(answer)
        else:
            st.info("👆 请先上传文档")

if __name__ == "__main__":
    main()
