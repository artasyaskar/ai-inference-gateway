"""
Alembic environment configuration.

This script configures Alembic for database migrations.
Uses synchronous engine as migrations don't need async.
"""

from logging.config import fileConfig

from sqlalchemy import engine_from_config, create_engine, pool

from alembic import context

# Import application models
from app.database import Base
from app.models.database_models import User, APIUsage, InferenceRequest, ModelCache, SystemMetrics
from app.config import settings

# this is the Alembic Config object
config = context.config

# Interpret the config file for Python logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Add your model's MetaData object here for 'autogenerate' support
target_metadata = Base.metadata

# Override database URL from settings - use sync URL for migrations
def get_database_url():
    """Get database URL from application settings (sync version)."""
    # Use the regular DATABASE_URL (should be postgresql:// not postgresql+asyncpg://)
    return settings.DATABASE_URL

config.set_main_option("sqlalchemy.url", get_database_url())


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode using synchronous engine."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
