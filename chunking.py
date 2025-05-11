from llama_index.core import Document
from llama_index.core.node_parser import SimpleNodeParser

def chunk_text(text: str, chunk_size: int = 200, chunk_overlap: int = 20) -> list:
    """
    Chunk the input text into smaller segments using SimpleNodeParser.
    
    Args:
        text (str): Input text to chunk.
        chunk_size (int): Maximum size of each chunk (in tokens). Default is 200.
        chunk_overlap (int): Overlap between chunks (in tokens). Default is 20.
    
    Returns:
        list: List of nodes (chunks) from llama_index.
    
    Raises:
        ValueError: If chunking fails or no chunks are created.
    """
    try:
        if not text:
            raise ValueError("Input text is empty")
        
        # Initialize parser
        parser = SimpleNodeParser.from_defaults(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Create document and parse into nodes
        document = Document(text=text)
        nodes = parser.get_nodes_from_documents([document])
        
        if not nodes:
            raise ValueError("No chunks created from text")
        
        return nodes
    except Exception as e:
        raise ValueError(f"Chunking failed: {e}")