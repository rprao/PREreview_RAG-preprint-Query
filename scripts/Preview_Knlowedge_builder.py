from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.node_parser import SentenceSplitter
from qdrant_client import QdrantClient
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import Settings
from scripts.doc_cleanup import clean_up_text
from llama_index.llms.groq import Groq
import os
from dotenv import load_dotenv
load_dotenv()


# Function to build the index
def PREreview_knowlegebase_index_builder(cleaned_documents):
    """The function uses the knowledge base, scraped from PREreview.org website, 
    to build a sentence-level index for use as one of the query engines.
    
    AIM: To resolve general queries about PREreview."""
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=3,
        window_metadata_key="window",
        original_text_metadata_key="original_text"
    )
    Settings.llm = Groq(model="llama3-70b-8192", api_key=os.getenv("GROQ_API_KEY"))
    Settings.embed_model = FastEmbedEmbedding(model_name="intfloat/multilingual-e5-large",cache_dir="D:/PREreviewBOT/multilingual_embed_cache")
    nodes = node_parser.get_nodes_from_documents(cleaned_documents)
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    qdrant_client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY
    )
    vector_store = QdrantVectorStore(client=qdrant_client, collection_name="multiligual_PREreview_Sentence_level")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    prereview_sentence_index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context
    )
    return prereview_sentence_index