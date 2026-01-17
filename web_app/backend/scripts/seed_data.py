import asyncio
from app.db.database import init_db, async_session
from app.core.security import get_password_hash
from app.db.models import User

async def seed_db():
    print("Initializing DB...")
    await init_db()
    
    async with async_session() as session:
        # Check if user exists
        existing = await session.get(User, 1)
        if existing:
            print("User already exists.")
            return

        print("Creating test user...")
        user = User(
            username="testuser",
            email="test@example.com",
            hashed_password=get_password_hash("test1234"),
            full_name="Test User",
            role="admin",
            is_active=True
        )
        session.add(user)
        await session.commit()
        print("✓ User created: testuser / test1234")

if __name__ == "__main__":
    asyncio.run(seed_db())
