"""
The single SQLAlchemy declarative base.

Memory tables (`app/memory/models.py`) and application-record tables
(`app/domain/models.py`) are deliberately separate modules but must share one
`Base`, because `Base.metadata` is what `create_all` and the migration scripts
walk. Splitting the base as well would produce two metadata registries and a
schema that is only ever half-created.
"""
from sqlalchemy.orm import declarative_base

Base = declarative_base()
