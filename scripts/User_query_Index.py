import nest_asyncio
nest_asyncio.apply()
from llama_index.core import VectorStoreIndex
from llama_index.core import Settings
from llama_index.llms.groq import Groq
from llama_index.embeddings.fastembed import FastEmbedEmbedding
import os
from llama_index.core.postprocessor import LongContextReorder
from llama_index.core.node_parser import UnstructuredElementNodeParser
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core import StorageContext
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import AutoMergingRetriever

Settings.llm=Groq(model="llama3-8b-8192", api_key=os.getenv("GROQ_API_KEY"))
Settings.embed_model = FastEmbedEmbedding(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",cache_dir="D:/PREreviewBOT/multilingual_embed_cache")
Settings.chunk_size = 1024
Settings.chunk_overlap = 20

def user_vector_engine(user_cleaned_docs,response_synthesizer):
    """Needs an input documents
    user_cleaned_docs: pdf
    
    If you need to run without router query here is an example
    
    reorder_response = user_query_engine.query("எனக்கு கட்டுரையின் விரிவான மற்றும் சுருக்கமான சுருக்கம் தேவை")
    display_response(reorder_response)
    
    """
    node_parser = UnstructuredElementNodeParser()
    nodes = node_parser.get_nodes_from_documents(user_cleaned_docs)

    docstore = SimpleDocumentStore()
    docstore.add_documents(nodes)
    storage_context = StorageContext.from_defaults(docstore=docstore)

    user_vector_index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        )
    base_retriever = user_vector_index.as_retriever(similarity_top_k=10)
    retriever = AutoMergingRetriever(base_retriever, storage_context)
    user_query_engine = RetrieverQueryEngine.from_args(retriever, response_synthesizer=response_synthesizer)
    
    return user_query_engine