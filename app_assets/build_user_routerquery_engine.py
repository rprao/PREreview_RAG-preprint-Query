import nest_asyncio
nest_asyncio.apply()
from llama_index.core.query_engine import ToolRetrieverRouterQueryEngine
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
)
from llama_index.core import VectorStoreIndex
from llama_index.core.objects import ObjectIndex
from llama_index.core.tools import QueryEngineTool
from llama_index.core import SimpleDirectoryReader
from scripts.selection_loader import link_to_pdf, BIN_FOLDER  # Import from your selection_loader.py
from scripts.doc_cleanup import clean_up_text
from scripts.User_query_Index import user_vector_engine
from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.core import PromptTemplate
from llama_index.core.query_engine import RetryQueryEngine
from llama_index.core.evaluation import RelevancyEvaluator
import os

def build_user_routerquery_engine(pdf_link):
    document = link_to_pdf(pdf_link)
    if document:
        pdf_path = os.path.join(BIN_FOLDER, 'File_for_RAG.pdf')
        with open(pdf_path, 'wb') as f:
            f.write(document)
            print("PDF downloaded successfully.")
    else:
        print("Failed to download the PDF.")
    user_documents = SimpleDirectoryReader(BIN_FOLDER).load_data()
    # clean the docs
    user_cleaned_docs = []
    for d in user_documents: 
        cleaned_text = clean_up_text(d.text)
        d.text = cleaned_text
        user_cleaned_docs.append(d)

    # Build a user query engine if the link is provided
    user_query_engine = user_vector_engine(user_cleaned_docs)
    emotion_stimuli_dict = {
        "01": "This is very important to my career.",
        "02": "You'd better be sure.",
        "03": "Stay focused and dedicated to your goals.Your consistent efforts will lead to outstanding achievements"
    }

    TREE_SUMMARIZE_PROMPT_TMPL = ("Context information from multiple sources is below, where some sources may or may not contain relevance scores.\n"
    "--------------------------------\n"
    "{context_str}\n"
    "--------------------------------\n"
    "Given the information from multiple sources and using only the information from these sources and not prior knowledge"
    "create a concise, accurate, and professional summary, adopting the tone of an academic professor, philosopher, or scientist.\n"
    "Consider the emotional stimuli when crafting your response, but do not explicitly mention or repeat them in the answer.\n"
    "If the information is insufficient, state that the question cannot be answered based on the given context.\n"
    "{emotion_stimuli_dict}\n"
    "Question: {query_str}\n"
    "Answer:")
    treesummarize = TreeSummarize(summary_template=PromptTemplate(TREE_SUMMARIZE_PROMPT_TMPL))
    user_vector_tool = QueryEngineTool.from_defaults(query_engine=user_query_engine)
    reorder_obj_index = ObjectIndex.from_objects([user_vector_tool],index_cls=VectorStoreIndex)
    user_routerquery_engine = ToolRetrieverRouterQueryEngine(reorder_obj_index.as_retriever(),summarizer= treesummarize)

    #self evaluating query engine

    query_response_evaluator = RelevancyEvaluator()
    user_source_query_engine = RetryQueryEngine(
        user_routerquery_engine, query_response_evaluator, max_retries=2,
    )
    return user_source_query_engine