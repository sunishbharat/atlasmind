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
                cur.execute("INSERT INTO items (annotation, jql embedding) VALUES (%s, %s %s);",
                            (chunk["annotation"], chunk["jql"], embed))

    logging.info("Embeddings Updated ...")
    
    
    
def document_proc_test(path:str|PathLike) -> list[str]:

    logger.info(f"{path=} : {page_chunks=}")
    embedconfig:EmbeddingsConfig = EmbeddingsConfig( model_name= model_1)

    doc = DocumentProcessor(embedconfig=embedconfig)
    content_list, model = doc.embeddings_generate(path=path, page_chunks=50)

    # Encode text documents into fixed-size vector embeddings using SentenceTransformer.
#    embed_list = model.encode(content_list)

    # Commit it into postgres vector db
#    pgVector_db_update(model=model, content_extract_list=content_list, embeddings_list=embed_list)


    # Test vector embeddings inference for similarity search.
    query = "find the number of bugs fixed with high piority in last 7 days"
    records = test_embeddings(query=query, model=model)
    return records

if __name__=="__main__":
    document_proc_test(path=r"./data/jql_queries_merged.pdf")
    logging.info("document_proc_test completed")