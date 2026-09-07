"""Database session and connection management."""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from src.config import settings
from src.db.models import Base

# Configure SQLite or PostgreSQL
connect_args = {}
if settings.DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(
    settings.DATABASE_URL,
    connect_args=connect_args,
    echo=False,
    pool_pre_ping=True
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


from sqlalchemy import text


def init_db():
    """Create all database tables and perform lightweight column migrations if needed."""
    Base.metadata.create_all(bind=engine)

    # Resilient column migration for SQLite
    with engine.connect() as conn:
        try:
            if settings.DATABASE_URL.startswith("sqlite"):
                cursor = conn.execute(text("PRAGMA table_info(relationships)"))
                existing_cols = {row[1] for row in cursor.fetchall()}
                new_cols = {
                    "decision_route": "VARCHAR(100)",
                    "review_reason": "VARCHAR(100)",
                    "comparison_snapshot": "JSON",
                    "extraction_confidence_a": "FLOAT DEFAULT 1.0",
                    "extraction_confidence_b": "FLOAT DEFAULT 1.0",
                }
                for col_name, col_type in new_cols.items():
                    if col_name not in existing_cols:
                        conn.execute(text(f"ALTER TABLE relationships ADD COLUMN {col_name} {col_type}"))
                conn.commit()
        except Exception as e:
            print(f"[init_db Migration Note] {e}")


def get_db():
    """Dependency for obtaining a database session."""
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()
