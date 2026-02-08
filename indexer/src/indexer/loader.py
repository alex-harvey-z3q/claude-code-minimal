import json
from collections.abc import Iterable

import boto3
from llama_index.core import Document

from indexer.settings import PARSED_BUCKET, PARSED_PREFIX


def _iter_s3_keys(bucket: str, prefix: str) -> Iterable[str]:
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".json"):
                yield key


def load_documents() -> list[Document]:
    s3 = boto3.client("s3")
    docs: list[Document] = []

    for key in _iter_s3_keys(PARSED_BUCKET, PARSED_PREFIX):
        obj = s3.get_object(Bucket=PARSED_BUCKET, Key=key)
        data = json.loads(obj["Body"].read().decode("utf-8"))

        text = data.get("text", "")
        if not text.strip():
            continue

        metadata = dict(data.get("metadata") or {})
        metadata["s3_key"] = key

        docs.append(Document(text=text, doc_id=data.get("doc_id"), metadata=metadata))

    return docs
