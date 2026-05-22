import asyncio
import os

from dotenv import load_dotenv
from obstore.store import ObjectStore, from_url

ICECAT_PREFIX = "icecat-"
LIST_CHUNK_SIZE = int(os.getenv("CEPH_CLEAN_LIST_CHUNK_SIZE", "1000"))
MAX_CONCURRENT_PREFIXES = int(os.getenv("CEPH_CLEAN_CONCURRENCY", "8"))


async def _delete_prefix(store: ObjectStore, prefix: str) -> int:
    deleted = 0
    async for batch in store.list(prefix=prefix, chunk_size=LIST_CHUNK_SIZE):
        paths = [obj["path"] for obj in batch]
        if paths:
            await store.delete_async(paths)
            deleted += len(paths)
    return deleted


async def clean_store(store: ObjectStore) -> int:
    root_listing = await store.list_with_delimiter_async()
    prefixes = sorted(
        prefix
        for prefix in root_listing["common_prefixes"]
        if prefix.startswith(ICECAT_PREFIX)
    )
    root_objects = [
        obj["path"]
        for obj in root_listing["objects"]
        if obj["path"].startswith(ICECAT_PREFIX)
    ]

    deleted = 0
    if root_objects:
        await store.delete_async(root_objects)
        deleted += len(root_objects)

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_PREFIXES)

    async def delete_with_limit(prefix: str) -> int:
        async with semaphore:
            return await _delete_prefix(store, prefix)

    if prefixes:
        deleted += sum(
            await asyncio.gather(*(delete_with_limit(prefix) for prefix in prefixes))
        )

    return deleted


def build_store() -> ObjectStore:
    load_dotenv()

    access_key = os.getenv("CEPH_ACCESS_KEY_ID")
    secret_key = os.getenv("CEPH_SECRET_ACCESS_KEY")

    return from_url(
        "s3://intake-virtual-icechunk-store",
        config={
            "endpoint_url": "https://projects.pawsey.org.au",
            "access_key_id": access_key,
            "secret_access_key": secret_key,
        },
    )


def main() -> None:
    deleted = asyncio.run(clean_store(build_store()))
    print(f"Deleted {deleted} objects from the Ceph test store")


if __name__ == "__main__":
    main()
