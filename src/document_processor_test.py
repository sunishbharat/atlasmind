import logging
import os
import psycopg2
from urllib.parse import urlparse
from os import PathLike
from document_processor import DocumentProcessor
from dconfig import EmbeddingsConfig
from pgvector_client import PGVectorClient, PGVectorConfig
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model_1 = "BAAI/bge-small-en-v1.5"

page_chunks = 50

######################################
# get_pgConfig_env
#
######################################
def get_pgConfig_env()-> PGVectorConfig:
    url = urlparse(os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:postgres@localhost:5432/jql_vectordb"
    ))
    pgConfig = PGVectorConfig(
        database=url.path.lstrip("/"),   # jql_vectordb
        user=url.username,               # postgres
        password=url.password,           # postgres
        host=url.hostname,               # pgvector
        port=url.port,                   # 5432
    )
    return pgConfig

##########################################
# Test the embeddings with a user query
##########################################
def test_embeddings(query:str, model:SentenceTransformer) -> list[tuple]:
    query_emb = model.encode(query, normalize_embeddings=True)
    
    sql = """
        SELECT id, text, embedding <-> %s AS distance
        FROM items
        ORDER BY embedding <-> %s
        LIMIT 5;
        """

    logging.info(f"SQL Query = {sql} ")

    with PGVectorClient(get_pgConfig_env()) as pgclient:
        with pgclient.cursor() as cur:
            cur.execute(sql, (query_emb, query_emb))
            rows = cur.fetchall()

    for id_, text, dist in rows:
        logging.info("*"*40)
        logging.info(f"{id_=}, {dist=}, ->\n {text}")

    return rows


##########################################
# Test the embeddings for JQL annotations
##########################################
def test_embeddings_jql(query: str, model: SentenceTransformer) -> tuple[list[tuple], SentenceTransformer]:
    query_emb = model.encode(query, normalize_embeddings=True)

    sql = """
        SELECT id, annotation, jql, embedding <-> %s AS distance
        FROM items
        ORDER BY embedding <-> %s
        LIMIT 5;
        """

    logging.info(f"SQL Query = {sql}")

    with PGVectorClient(get_pgConfig_env()) as pgclient:
        with pgclient.cursor() as cur:
            cur.execute(sql, (query_emb, query_emb))
            rows = cur.fetchall()

    for id_, annotation, jql, dist in rows:
        logging.info("*" * 40)
        logging.info(f"{id_=}, {dist=}, ->\n annotation: {annotation}\n jql: {jql}")

    return rows, model

######################################
# _ensure_extension
# Must run before PGVectorClient — register_vector() requires the vector
# type to already exist in the DB at connect time.
######################################
def _ensure_extension(pgConfig: PGVectorConfig) -> None:
    conn = psycopg2.connect(
        database=pgConfig.database, user=pgConfig.user,
        password=pgConfig.password, host=pgConfig.host, port=pgConfig.port,
    )
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.close()


######################################
# pgVector_db_update
#
######################################
def pgVector_db_update(model:SentenceTransformer ,content_extract_list:list, embeddings_list):

    embd_dim = model.get_sentence_embedding_dimension()
    pgConfig  = get_pgConfig_env()
    _ensure_extension(pgConfig)   # create extension before PGVectorClient registers vector type

    with PGVectorClient(pgConfig) as pgclient:
        with pgclient.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS ITEMS")
            cur.execute(f"""CREATE TABLE IF NOT EXISTS items (
            id bigserial PRIMARY KEY, annotation TEXT, jql TEXT, embedding vector({embd_dim}));""")

    with PGVectorClient(pgConfig) as pgclient:
        for chunk, embed in zip(content_extract_list, embeddings_list):
            with pgclient.cursor() as cur:
                cur.execute("INSERT INTO items (annotation, jql, embedding) VALUES (%s, %s, %s);",
                            (chunk["comment"], chunk["jql"], embed))

    logging.info("Embeddings Updated ...")


######################################
# update_pgvector_from_annotations
#
# Encodes the 'comment' field of each annotation pair produced by
# parse_jql_annotations(), then stores (comment, jql, embedding) in pgvector.
######################################
def update_pgvector_from_annotations(
    pairs: list[dict[str, str]],
    model_name: str = model_1,
) -> SentenceTransformer:
    """Embed annotation comments and upsert (comment, jql, embedding) rows into pgvector.

    Args:
        pairs: Output of parse_jql_annotations() — list of {"comment": ..., "jql": ...} dicts.
        model_name: SentenceTransformer model used for encoding. Defaults to module-level model_1.

    Returns:
        The SentenceTransformer model used for encoding (reuse for similarity search).
    """
    if not pairs:
        logging.warning("update_pgvector_from_annotations: empty pairs list, nothing to store.")
        return

    embedconfig = EmbeddingsConfig(model_name=model_name)
    processor = DocumentProcessor(embedconfig=embedconfig)

    comments = [p["comment"] for p in pairs]
    embeddings = processor._model.encode(comments, batch_size=32, show_progress_bar=True, normalize_embeddings=True)

    logging.info("Encoding complete (%d vectors). Updating pgvector ...", len(embeddings))
    pgVector_db_update(model=processor._model, content_extract_list=pairs, embeddings_list=embeddings)
    return processor._model

    return processor._model
    
    
    
def document_proc_test(path:str|PathLike) -> list[str]:

    logger.info(f"{path=} : {page_chunks=}")
    embedconfig:EmbeddingsConfig = EmbeddingsConfig( model_name= model_1)

    doc = DocumentProcessor(embedconfig=embedconfig)
    content_list, model = doc.embeddings_generate(path=path, page_chunks=50)

    # Encode text documents into fixed-size vector embeddings using SentenceTransformer.
    embed_list = model.encode(content_list)

    # Commit it into postgres vector db
    pgVector_db_update(model=model, content_extract_list=content_list, embeddings_list=embed_list)


    # Test vector embeddings inference for similarity search.
    query = "find the number of bugs fixed with high piority in last 7 days"
    records = test_embeddings(query=query, model=model)
    return records, model

from jql_annotation_parser import parse_jql_annotations
def jql_annotation_test():
    logging.info("jql_annotation_test started")
    pairs = parse_jql_annotations(path=r"./data/new_Format_jql_annotated.md")
    model = update_pgvector_from_annotations(pairs=pairs)
    logging.info("jql_annotation_test completed")
    return model

if __name__=="__main__":
    # document_proc_test(path=r"./data/jql_queries_merged.pdf")
    model = jql_annotation_test()
    records, model = test_embeddings_jql(
        query="find the number of bugs fixed with high piority in last 7 days", model=model)
    logging.info("document_proc_test completed")