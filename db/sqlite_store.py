"""
db/sqlite_store.py

Async SQLite wrapper for article metadata.
Stores article info, dedup hashes, and processing state.
"""

import aiosqlite
import uuid
from config import SQLITE_DB_PATH, DATA_DIR


async def init_db():
    """Create the articles table if it doesn't exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    async with aiosqlite.connect(SQLITE_DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS articles (
                id              TEXT PRIMARY KEY,
                url             TEXT UNIQUE,
                title           TEXT NOT NULL,
                source          TEXT,
                published_at    DATETIME,
                fetched_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
                category        TEXT,
                summary         TEXT,
                content         TEXT,
                content_hash    TEXT,
                embedding_id    TEXT,
                story_cluster   TEXT,
                is_processed    BOOLEAN DEFAULT 0
            )
        """)
        await db.commit()
        print("[sqlite] Database initialized")


async def article_exists(url: str) -> bool:
    """Check if an article with this URL already exists."""
    async with aiosqlite.connect(SQLITE_DB_PATH) as db:
        cursor = await db.execute(
            "SELECT 1 FROM articles WHERE url = ?", (url,)
        )
        row = await cursor.fetchone()
        return row is not None


async def content_hash_exists(content_hash: str) -> bool:
    """Check if an article with this content hash already exists."""
    async with aiosqlite.connect(SQLITE_DB_PATH) as db:
        cursor = await db.execute(
            "SELECT 1 FROM articles WHERE content_hash = ?", (content_hash,)
        )
        row = await cursor.fetchone()
        return row is not None


async def insert_article(article: dict) -> str:
    """
    Insert a new article and return its UUID.

    Expected keys: url, title, source, published_at, category,
                   summary, content, content_hash
    """
    article_id = str(uuid.uuid4())

    async with aiosqlite.connect(SQLITE_DB_PATH) as db:
        await db.execute(
            """
            INSERT INTO articles (id, url, title, source, published_at,
                                  category, summary, content, content_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                article_id,
                article.get("url"),
                article.get("title"),
                article.get("source"),
                article.get("published_at"),
                article.get("category"),
                article.get("summary"),
                article.get("content"),
                article.get("content_hash"),
            ),
        )
        await db.commit()

    return article_id


async def mark_processed(article_id: str):
    """Mark an article as fully processed (embedded + stored)."""
    async with aiosqlite.connect(SQLITE_DB_PATH) as db:
        await db.execute(
            "UPDATE articles SET is_processed = 1 WHERE id = ?",
            (article_id,),
        )
        await db.commit()


async def get_unprocessed() -> list[dict]:
    """Get all articles that haven't been embedded yet."""
    async with aiosqlite.connect(SQLITE_DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM articles WHERE is_processed = 0"
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]


async def get_article(article_id: str) -> dict | None:
    """Retrieve a single article by ID."""
    async with aiosqlite.connect(SQLITE_DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM articles WHERE id = ?", (article_id,)
        )
        row = await cursor.fetchone()
        return dict(row) if row else None


async def get_article_count() -> int:
    """Return total number of articles stored."""
    async with aiosqlite.connect(SQLITE_DB_PATH) as db:
        cursor = await db.execute("SELECT COUNT(*) FROM articles")
        row = await cursor.fetchone()
        return row[0]


async def update_story_cluster(article_id: str, story_cluster: str):
    """Update the story_cluster field for an article."""
    async with aiosqlite.connect(SQLITE_DB_PATH) as db:
        await db.execute(
            "UPDATE articles SET story_cluster = ? WHERE id = ?",
            (story_cluster, article_id),
        )
        await db.commit()

