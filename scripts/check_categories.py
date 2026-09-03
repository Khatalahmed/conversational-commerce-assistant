import asyncio

import sys
from pathlib import Path

# Dev script lives in scripts/, so the project root must be on sys.path
# for `import app` to resolve when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from app.db.connection import connect_to_mongo, close_mongo_connection
from app.repos import categories_repo


async def main():
    await connect_to_mongo()

    categories = await categories_repo.get_all_categories()
    print(f"{len(categories)} categories found:\n")
    for c in categories:
        print(f"{c['name']}: {', '.join(c['subCategories'])}")

    await close_mongo_connection()


asyncio.run(main())