import sys
sys.path.insert(0, '.')

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.models.database_models import User, UserTier

engine = create_engine("postgresql://postgres@localhost:5432/inference_gateway")
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def create_user():
    db = SessionLocal()
    try:
        user = User(
            api_key="ak_test_yourapikey123456",
            tier=UserTier.PRO,
            is_active=True
        )
        db.add(user)
        db.commit()
        print(f"✅ Created user: {user.id}")
        print(f"🔑 API Key: {user.api_key}")
    finally:
        db.close()

if __name__ == "__main__":
    create_user()