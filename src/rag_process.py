import logging

from document_processor import DocumentProcessor
from dconfig import EmbeddingsConfig
from pgvector_client import PGVectorClient, PGVectorConfig

logger = logging.getLogger(__name__)


def generate_embeddings(path: str):
    config = EmbeddingsConfig(model_name="BAAI/bge-base-en-v1.5")
    processor = DocumentProcessor(embedconfig=config)

    content_list, model = processor.embeddings_generate(path=path)
    embeddings = model.encode(content_list, batch_size=32, show_progress_bar=True)
    return embeddings, model


def store_embeddings(embeddings, model):
    pg_config = PGVectorConfig(host="localhost", database="jql_vectordb")
    with PGVectorClient(pg_config) as client:
        with client.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS jql_vector")
            cur.execute(f"CREATE TABLE IF NOT EXISTS items (id bigserial PRIMARY KEY, text TEXT, embedding vector({model.get_sentence_embedding_dimension()}))")

    for chunk, embed in zip(content_list, embeddings):
        with client.cursor() as cur:
            cur.execute("INSERT INTO items (text, embedding) VALUES (%s, %s)", (chunk, embed))


def search_embeddings(query: str, model):
    query_vec = model.encode(query, normalize_embeddings=True)
    pg_config = PGVectorConfig(host="localhost", database="jql_vectordb")
    with PGVectorClient(pg_config) as client:
        with client.cursor() as cur:
            cur.execute("""
                SELECT id, text, embedding <-> %s AS distance
            FROM items ORDER BY distance LIMIT 2
        """, (query_vec, query_vec))
        results = cur.fetchall()

    for id_, text, dist in results:
        logger.info("[%.4f] %s", dist, text[:100])
