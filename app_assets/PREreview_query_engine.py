import nest_asyncio
nest_asyncio.apply()
from llama_index.core.query_engine import ToolRetrieverRouterQueryEngine
from llama_index.core import VectorStoreIndex
from llama_index.core import Settings
from llama_index.core.objects import ObjectIndex
from llama_index.core.tools import QueryEngineTool
from scripts.PREreview_sentence_index_query_engine import PREreview_sentence_index_query_engine
from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.core import PromptTemplate
from llama_index.core.query_engine import RetryQueryEngine
from llama_index.core.evaluation import RelevancyEvaluator

def PREreview_engine():
    Prereview_query_engine = PREreview_sentence_index_query_engine()
    #call the Quadrant containing the knowledge of PREreview
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
    Prereview_vector_tool = QueryEngineTool.from_defaults(query_engine=Prereview_query_engine)
    Prereview_reorder_obj_index = ObjectIndex.from_objects([Prereview_vector_tool],index_cls=VectorStoreIndex)
    Prereview_routerquery_engine = ToolRetrieverRouterQueryEngine(Prereview_reorder_obj_index.as_retriever(),summarizer= treesummarize)
    query_response_evaluator = RelevancyEvaluator()
    Prereview_retry_source_query_engine = RetryQueryEngine(
        Prereview_routerquery_engine, query_response_evaluator, max_retries= 2,
    )
    return Prereview_retry_source_query_engine