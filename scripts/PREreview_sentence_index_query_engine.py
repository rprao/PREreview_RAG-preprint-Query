from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from qdrant_client import QdrantClient
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import Settings
from llama_index.llms.groq import Groq
import os
from dotenv import load_dotenv
load_dotenv()

def PREreview_sentence_index_query_engine():
    #define settings although 
    Settings.llm = Groq(model="llama3-70b-8192", api_key=os.getenv("GROQ_API_KEY"))
    Settings.embed_model =  FastEmbedEmbedding(model_name="intfloat/multilingual-e5-large",cache_dir="D:/PREreviewBOT/multilingual_embed_cache")
    #initiate Qdrant clients                                               
    QDRANT_URL= os.getenv("QDRANT_URL") 
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    qdrant_client = QdrantClient(
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY,
                )
    #rebuild from Qdrant_store
    vector_store =QdrantVectorStore(client=qdrant_client,collection_name="multiligual_PREreview_Sentence_level")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    prereview_sentence_index = VectorStoreIndex.from_vector_store(vector_store=vector_store,storage_context=storage_context)

    return prereview_sentence_index.as_query_engine(similarity_top_k=5,node_postprocessors=[MetadataReplacementPostProcessor(target_metadata_key="window")])
   
