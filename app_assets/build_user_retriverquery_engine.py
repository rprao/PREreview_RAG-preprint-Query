import nest_asyncio
nest_asyncio.apply()
from llama_index.core.query_engine import ToolRetrieverRouterQueryEngine
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
)
from llama_index.core import get_response_synthesizer
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

def build_user_retriverquery_engine(pdf_link):
    document = link_to_pdf(pdf_link)
    if os.path.exists(BIN_FOLDER) and os.listdir(BIN_FOLDER):  # Check if folder exists and is not empty
        for filename in os.listdir(BIN_FOLDER):
            file_path = os.path.join(BIN_FOLDER, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):  # Only remove files, not directories
                    os.unlink(file_path)  # Delete the file or symbolic link
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Remove directory if needed
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
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
    emotion_stimuli_dict = {
        "01": "This is very important to my career.",
        "02": "You'd better be sure.",
        "03": "Stay focused and dedicated to your goals.Your consistent efforts will lead to outstanding achievements"
    }
    TREE_SUMMARIZE_PROMPT_TMPL = ("Context information from multiple sources is below, where some sources may or may not contain relevance scores.\n"
    "--------------------------------\n"
    "{context_str}\n"
    "--------------------------------\n"
    "Given the information from multiple sources and using only the information from these sources and not prior knowledge\n"
    "create a concise, accurate, and professional summary, adopting the tone of an academic professor, philosopher, or scientist.\n"
    "Consider the emotional stimuli when crafting your response, but do not explicitly mention or repeat them in the answer.\n"
    "If the information is insufficient, state that the question cannot be answered based on the given context.\n"
    "When responding, **always provide a fresh answer** instead of repeating the original response. Do not say **I would repeat the original answer.**\n"
    "{emotion_stimuli_dict}\n"
    "Question: {query_str}\n"
    "Answer:")
    treesummarize = TreeSummarize(summary_template=PromptTemplate(TREE_SUMMARIZE_PROMPT_TMPL))
    response_synthesizer = get_response_synthesizer(summary_template=treesummarize,use_async=True,streaming=True)
    user_source_query_engine = user_vector_engine(user_cleaned_docs,response_synthesizer)
    
    return user_source_query_engine