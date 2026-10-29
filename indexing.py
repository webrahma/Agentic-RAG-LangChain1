from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from fileloder import read_multiple_pdfs

# تحميل مفتاح GOOGLE_API_KEY من ملف .env
load_dotenv()

# 1. إعداد نموذج الـ Embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# 2. إنشاء قاعدة البيانات المتجهة في الذاكرة
vectore_store = InMemoryVectorStore(embeddings)

# 3. مسارات ملفات الـ PDF
DOC_PATHS = [
    "pdfs/PartA.pdf",
    "pdfs/PartB.pdf"
]

# 4. قراءة الملفات
docs = read_multiple_pdfs(DOC_PATHS)

# 5. تقسيم النصوص إلى أجزاء صغيرة (Chunks)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
splits = text_splitter.split_documents(docs)

# 6. تخزين الأجزاء في الـ Vector Store
vectore_store.add_documents(documents=splits)