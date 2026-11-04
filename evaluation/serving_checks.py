"""Destructive only to a deliberately empty test service: real HTTP update/worker checks."""

import asyncio

import httpx


async def check_service(url: str, token: str, workers: int = 2) -> dict:
    headers = {"Authorization": "Bearer " + token, "Connection": "close"}
    async with httpx.AsyncClient(base_url=url, headers=headers, trust_env=False, timeout=30) as client:
        initial = (await client.get("/health")).json()
        if initial["count"] != 0:
            raise ValueError("Consistency checks require a new empty test service")
        created = await client.post("/documents", json={"documents": [{"id": "seed", "text": "seed document"}]})
        created.raise_for_status()
        first_version = created.json()["snapshot_version"]

        async def publish(i):
            response = await client.post(
                "/documents", json={"documents": [{"id": f"doc-{i}", "text": f"unique item {i}"}]}
            )
            response.raise_for_status()
            return response.json()

        async def read_during_updates():
            samples = []
            for _ in range(40):
                response = await client.post("/retrieve", json={"query": "unique item", "k": 100})
                response.raise_for_status()
                samples.append(response.json())
                await asyncio.sleep(0.001)
            return samples

        updates, concurrent_reads = await asyncio.gather(
            asyncio.gather(*[publish(i) for i in range(12)]), read_during_updates()
        )
        counts = {first_version: 1, **{u["snapshot_version"]: u["count"] for u in updates}}
        for read in concurrent_reads:
            assert len(read["results"]) == counts[read["snapshot_version"]], "Mixed-version query result"
            assert len({r["id"] for r in read["results"]}) == len(read["results"])
        assert sorted(u["count"] for u in updates) == list(range(2, 14)), "Concurrent ingestion lost an update"
        health = (await client.get("/health")).json()
        assert health["count"] == 13
        expected_ids = {"seed", *[f"doc-{i}" for i in range(12)]}
        pids = set()
        for _ in range(30):
            responses = await asyncio.gather(
                *[client.post("/retrieve", json={"query": "unique item", "k": 100}) for _ in range(8)]
            )
            for response in responses:
                response.raise_for_status()
                result = response.json()
                assert result["snapshot_version"] == health["version"]
                assert {r["id"] for r in result["results"]} == expected_ids
                pids.add(result["pid"])
            if len(pids) == workers:
                break
            await asyncio.sleep(0.05)
        assert len(pids) == workers, "Did not verify reads from every configured worker"
        conflict = await client.post(
            "/documents", json={"documents": [{"id": "stale", "text": "stale"}], "expected_version": first_version}
        )
        assert conflict.status_code == 409
        return {
            "count": 13,
            "version": health["version"],
            "worker_pids": sorted(pids),
            "concurrent_updates": 12,
            "queries_during_updates": len(concurrent_reads),
            "stale_update_status": conflict.status_code,
        }


async def check_restart(url: str, token: str, expected: dict) -> dict:
    async with httpx.AsyncClient(
        base_url=url, headers={"Authorization": "Bearer " + token, "Connection": "close"}, trust_env=False, timeout=30
    ) as client:
        health = (await client.get("/health")).json()
        assert health["version"] == expected["version"] and health["count"] == expected["count"]
        response = await client.post("/retrieve", json={"query": "seed document", "k": 1})
        response.raise_for_status()
        assert response.json()["results"][0]["id"] == "seed"
        pids = set()
        for _ in range(30):
            responses = await asyncio.gather(*[client.get("/health") for _ in range(8)])
            for item in responses:
                item.raise_for_status()
                state = item.json()
                assert state["version"] == expected["version"] and state["count"] == expected["count"]
                pids.add(state["pid"])
            if len(pids) == len(expected["worker_pids"]):
                break
            await asyncio.sleep(0.05)
        assert len(pids) == len(expected["worker_pids"])
        return {
            "restored_worker_pids": sorted(pids),
            "restored_version": health["version"],
            "restored_count": health["count"],
            "query_after_restart": "passed",
        }
