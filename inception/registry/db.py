"""SQLite storage for the Gene Registry.

Stores agent cards, capabilities, and mating logs for
discovery and matchmaking.
"""

from __future__ import annotations

from datetime import datetime, timezone

import aiosqlite

from inception.a2a.models import GeneticAgentCard, RegistryEntry

_SCHEMA = """
CREATE TABLE IF NOT EXISTS agents (
    agent_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    url TEXT NOT NULL,
    card_json TEXT NOT NULL,
    registered_at TEXT NOT NULL,
    last_seen TEXT NOT NULL,
    best_fitness REAL DEFAULT 0.0,
    generation INTEGER DEFAULT 0,
    mating_count INTEGER DEFAULT 0,
    offspring_count INTEGER DEFAULT 0,
    active INTEGER DEFAULT 1
);

CREATE TABLE IF NOT EXISTS capabilities (
    agent_id TEXT NOT NULL,
    capability TEXT NOT NULL,
    strength REAL NOT NULL,
    PRIMARY KEY (agent_id, capability),
    FOREIGN KEY (agent_id) REFERENCES agents(agent_id)
);

CREATE TABLE IF NOT EXISTS mating_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    requester_id TEXT NOT NULL,
    responder_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    accepted INTEGER NOT NULL,
    offspring_id TEXT,
    task_description TEXT
);
"""


class RegistryDB:
    """Async SQLite database for the Gene Registry."""

    def __init__(self, db_path: str = "gene_registry.db"):
        self.db_path = db_path
        self._conn: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        self._conn = await aiosqlite.connect(self.db_path)
        await self._conn.executescript(_SCHEMA)
        await self._conn.commit()

    async def close(self) -> None:
        if self._conn:
            await self._conn.close()
            self._conn = None

    def _get_conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        return self._conn

    async def register_agent(self, card: GeneticAgentCard) -> None:
        conn = self._get_conn()
        now = datetime.now(timezone.utc).isoformat()
        card_json = card.model_dump_json()

        await conn.execute(
            """INSERT OR REPLACE INTO agents
               (agent_id, name, url, card_json, registered_at, last_seen,
                best_fitness, generation, mating_count, offspring_count, active)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, 0, 1)""",
            (card.agent_id, card.name, card.url, card_json, now, now,
             card.best_fitness, card.generation),
        )

        # Update capabilities
        await conn.execute(
            "DELETE FROM capabilities WHERE agent_id = ?", (card.agent_id,)
        )
        for cap, strength in card.capabilities.items():
            await conn.execute(
                "INSERT INTO capabilities (agent_id, capability, strength) VALUES (?, ?, ?)",
                (card.agent_id, cap, strength),
            )

        await conn.commit()

    async def update_agent(self, card: GeneticAgentCard) -> None:
        conn = self._get_conn()
        now = datetime.now(timezone.utc).isoformat()
        card_json = card.model_dump_json()

        await conn.execute(
            """UPDATE agents SET name=?, url=?, card_json=?, last_seen=?,
               best_fitness=?, generation=? WHERE agent_id=?""",
            (card.name, card.url, card_json, now,
             card.best_fitness, card.generation, card.agent_id),
        )

        await conn.execute(
            "DELETE FROM capabilities WHERE agent_id = ?", (card.agent_id,)
        )
        for cap, strength in card.capabilities.items():
            await conn.execute(
                "INSERT INTO capabilities (agent_id, capability, strength) VALUES (?, ?, ?)",
                (card.agent_id, cap, strength),
            )

        await conn.commit()

    async def deactivate_agent(self, agent_id: str) -> None:
        conn = self._get_conn()
        await conn.execute(
            "UPDATE agents SET active = 0 WHERE agent_id = ?", (agent_id,)
        )
        await conn.commit()

    async def get_agent(self, agent_id: str) -> RegistryEntry | None:
        conn = self._get_conn()
        async with conn.execute(
            "SELECT card_json, registered_at, last_seen, mating_count, offspring_count "
            "FROM agents WHERE agent_id = ? AND active = 1",
            (agent_id,),
        ) as cursor:
            row = await cursor.fetchone()
            if not row:
                return None
            card = GeneticAgentCard.model_validate_json(row[0])
            return RegistryEntry(
                agent_card=card,
                registered_at=row[1],
                last_seen=row[2],
                mating_count=row[3],
                offspring_count=row[4],
            )

    async def search_by_capability(
        self, capability: str, min_strength: float = 0.0, limit: int = 20
    ) -> list[RegistryEntry]:
        conn = self._get_conn()
        async with conn.execute(
            """SELECT DISTINCT a.card_json, a.registered_at, a.last_seen,
                      a.mating_count, a.offspring_count
               FROM agents a
               JOIN capabilities c ON a.agent_id = c.agent_id
               WHERE c.capability LIKE ? AND c.strength >= ? AND a.active = 1
               ORDER BY c.strength DESC
               LIMIT ?""",
            (f"%{capability}%", min_strength, limit),
        ) as cursor:
            results = []
            async for row in cursor:
                card = GeneticAgentCard.model_validate_json(row[0])
                results.append(RegistryEntry(
                    agent_card=card,
                    registered_at=row[1],
                    last_seen=row[2],
                    mating_count=row[3],
                    offspring_count=row[4],
                ))
            return results

    async def get_all_active(self, limit: int = 100) -> list[RegistryEntry]:
        conn = self._get_conn()
        async with conn.execute(
            """SELECT card_json, registered_at, last_seen, mating_count, offspring_count
               FROM agents WHERE active = 1 ORDER BY best_fitness DESC LIMIT ?""",
            (limit,),
        ) as cursor:
            results = []
            async for row in cursor:
                card = GeneticAgentCard.model_validate_json(row[0])
                results.append(RegistryEntry(
                    agent_card=card,
                    registered_at=row[1],
                    last_seen=row[2],
                    mating_count=row[3],
                    offspring_count=row[4],
                ))
            return results

    async def log_mating(
        self,
        requester_id: str,
        responder_id: str,
        accepted: bool,
        offspring_id: str = "",
        task_description: str = "",
    ) -> None:
        conn = self._get_conn()
        now = datetime.now(timezone.utc).isoformat()
        await conn.execute(
            """INSERT INTO mating_log
               (requester_id, responder_id, timestamp, accepted, offspring_id, task_description)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (requester_id, responder_id, now, int(accepted), offspring_id, task_description),
        )

        if accepted:
            await conn.execute(
                "UPDATE agents SET mating_count = mating_count + 1 WHERE agent_id IN (?, ?)",
                (requester_id, responder_id),
            )
            if offspring_id:
                await conn.execute(
                    "UPDATE agents SET offspring_count = offspring_count + 1 WHERE agent_id IN (?, ?)",
                    (requester_id, responder_id),
                )

        await conn.commit()

    async def get_leaderboard(self, limit: int = 20) -> list[RegistryEntry]:
        conn = self._get_conn()
        async with conn.execute(
            """SELECT card_json, registered_at, last_seen, mating_count, offspring_count
               FROM agents WHERE active = 1
               ORDER BY best_fitness DESC LIMIT ?""",
            (limit,),
        ) as cursor:
            results = []
            async for row in cursor:
                card = GeneticAgentCard.model_validate_json(row[0])
                results.append(RegistryEntry(
                    agent_card=card,
                    registered_at=row[1],
                    last_seen=row[2],
                    mating_count=row[3],
                    offspring_count=row[4],
                ))
            return results
