"""Migration: Add content_hash column and unique index to documents table."""
import hashlib
from pathlib import Path
from sqlalchemy import text
from sqlalchemy.engine import Engine


def migrate_add_content_hash(engine: Engine):
    """Adds content_hash VARCHAR(64) and unique index to documents table if missing."""
    is_sqlite = engine.url.drivername.startswith("sqlite")
    
    with engine.begin() as conn:
        try:
            if is_sqlite:
                cursor = conn.execute(text("PRAGMA table_info(documents)"))
                existing_cols = {row[1] for row in cursor.fetchall()}
                if "content_hash" not in existing_cols:
                    conn.execute(text("ALTER TABLE documents ADD COLUMN content_hash VARCHAR(64)"))
                conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS ix_documents_content_hash ON documents(content_hash)"))
            else:
                conn.execute(text("ALTER TABLE documents ADD COLUMN IF NOT EXISTS content_hash VARCHAR(64)"))
                conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS ix_documents_content_hash ON documents(content_hash)"))
        except Exception as e:
            print(f"[Migration Note] Error adding content_hash column/index: {e}")

    # Backfill content_hash for any existing documents with NULL hash if file is accessible
    with engine.begin() as conn:
        try:
            rows = conn.execute(text("SELECT id, file_path FROM documents WHERE content_hash IS NULL")).fetchall()
            for doc_id, file_path in rows:
                if file_path and Path(file_path).is_file():
                    try:
                        h = hashlib.sha256(Path(file_path).read_bytes()).hexdigest()
                        # Only backfill if this content_hash is not already present on another document
                        conflict = conn.execute(
                            text("SELECT id FROM documents WHERE content_hash = :h AND id != :doc_id"),
                            {"h": h, "doc_id": doc_id}
                        ).first()
                        if not conflict:
                            conn.execute(
                                text("UPDATE documents SET content_hash = :h WHERE id = :doc_id"),
                                {"h": h, "doc_id": doc_id}
                            )
                    except Exception as err:
                        print(f"[Migration Note] Could not backfill hash for doc {doc_id}: {err}")
        except Exception as e:
            print(f"[Migration Note] Error during backfill: {e}")


if __name__ == "__main__":
    from src.db.session import engine
    print("Running migration: add content_hash to documents...")
    migrate_add_content_hash(engine)
    print("Migration complete.")
