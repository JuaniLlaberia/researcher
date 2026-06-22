from typing import List, Dict, Any

from src.db.vector_store import VectorStore

def serach_vector_db(main_query: str, queries: List[str]) -> List[Dict[str, Any]]:
    """
    Search in vector db using a main_query and query variants (queries).

    Args:
        main_query (str): Main query to search by and to use for reranking.
        queries (List[str]): List of query variants to search by to obtain deeper results.    
    Returns:
        List[Dict[str, Any]]: {id, text, metadata, score} sorted best first.
    """
    vector_store = VectorStore()
    results = vector_store.retrieve_documents(main_query=main_query,
                                              queries=queries)
    
    return results  