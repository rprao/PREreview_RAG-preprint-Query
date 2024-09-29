import nest_asyncio
import logging
import sys

from llama_index.core.query_engine import ToolRetrieverRouterQueryEngine
from llama_index.core import VectorStoreIndex
from llama_index.core.objects import ObjectIndex
from llama_index.core.tools import QueryEngineTool
from llama_index.core import PromptTemplate
from llama_index.core.response_synthesizers import TreeSummarize
from scripts.PREreview_sentence_index_query_engine import PREreview_sentence_index_query_engine

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Set up logging to print to standard output
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

def PREreview_Router_query_engine():
    """
    Function to create and return a Router Query Engine for PREreview
    """

    # Initialize the PREreview sentence index query engine
    PREreview_query_engine = PREreview_sentence_index_query_engine()

    # Create a QueryEngineTool for the PREreview vector tool
    PREreview_vector_tool = QueryEngineTool.from_defaults(
        query_engine=PREreview_query_engine,
        description="This is capable of answering queries from the PREreview knowledge base",
    )

    # Create an object index for the tool using VectorStoreIndex as the index class
    obj_index = ObjectIndex.from_objects(
        [PREreview_vector_tool],
        index_cls=VectorStoreIndex,
    )

    # Initialize TreeSummarize with asynchronous capabilities
    tree_summarize = TreeSummarize(use_async=True)

    # Create the PREreview Router Query Engine using the ObjectIndex as the retriever
    

    return ToolRetrieverRouterQueryEngine(retriever=obj_index.as_retriever(),summarizer=tree_summarize)
