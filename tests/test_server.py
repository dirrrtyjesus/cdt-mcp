import json
import math

import pytest
from mcp.client import Client
from mcp.shared.exceptions import MCPError

from cdt_mcp.server import create_server
from cdt_mcp.store import FieldStore


class FakeClock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def store(clock):
    return FieldStore(clock=clock)


@pytest.fixture
def server(store):
    return create_server(store)


async def call(client, tool, **args):
    result = await client.call_tool(tool, args)
    assert not result.is_error, result.content
    return result.structured_content


async def test_lists_all_tools_resources_prompts(server):
    async with Client(server) as client:
        tools = {t.name for t in (await client.list_tools()).tools}
        assert tools == {
            "cdt_create",
            "cdt_write",
            "cdt_read",
            "cdt_consensus",
            "cdt_list",
            "cdt_snapshot",
            "cdt_sync",
            "cdt_merge",
            "cdt_decay",
            "cdt_compact",
            "cdt_delete",
            "cdt_phase_of",
        }
        templates = {t.uri_template for t in (await client.list_resource_templates()).resource_templates}
        assert templates == {"cdt://field/{name}", "cdt://field/{name}/consensus"}
        resources = [str(r.uri) for r in (await client.list_resources()).resources]
        assert resources == ["cdt://fields"]
        prompts = {p.name for p in (await client.list_prompts()).prompts}
        assert prompts == {"cdt_reconcile"}
        assert "superposition" in (client.instructions or "")


async def test_annotations_are_declared(server):
    async with Client(server) as client:
        tools = {t.name: t for t in (await client.list_tools()).tools}
        assert tools["cdt_consensus"].annotations.read_only_hint is True
        assert tools["cdt_write"].annotations.destructive_hint is False
        assert tools["cdt_sync"].annotations.idempotent_hint is True
        assert tools["cdt_delete"].annotations.destructive_hint is True
        # Structured output schema is advertised
        assert "top_payload" in tools["cdt_consensus"].output_schema["properties"]


async def test_write_autocreates_and_consensus_reports_truth(server):
    async with Client(server) as client:
        w1 = await call(
            client, "cdt_write", field="decision", key="rebase", coherence=0.9, payload="Rebase", agent_id="a1"
        )
        w2 = await call(
            client, "cdt_write", field="decision", key="merge", coherence=0.7, payload="Merge", agent_id="a2"
        )
        assert w1["bin"] != w2["bin"]
        assert w1["weight"] == pytest.approx(0.9)
        c = await call(client, "cdt_consensus", field="decision", top_k=2)
        assert c["top_payload"] == "Rebase"
        assert c["share"] == pytest.approx(0.9 / 1.6)
        assert c["record_count"] == 2
        assert [b["payloads"][0]["payload"] for b in c["alternatives"]] == ["Rebase", "Merge"]
        assert c["payloads"][0]["agents"] == ["a1"]


async def test_disagreement_via_negative_value(server):
    async with Client(server) as client:
        await call(client, "cdt_write", field="d", key="rebase", coherence=1.0, payload="Rebase", agent_id="a1")
        await call(client, "cdt_write", field="d", key="merge", coherence=0.6, payload="Merge", agent_id="a2")
        await call(
            client, "cdt_write", field="d", key="rebase", value=-1.0, coherence=0.8, payload="Rebase", agent_id="a3"
        )
        c = await call(client, "cdt_consensus", field="d")
        assert c["top_payload"] == "Merge"
        assert c["contest_ratio"] == pytest.approx(1.6 / 2.4)
        assert c["contested"][0]["payloads"][0]["payload"] == "Rebase"
        r = await call(client, "cdt_read", field="d", key="rebase")
        assert r["amplitude"] == pytest.approx(0.2)
        assert sorted(r["payloads"][0]["agents"]) == ["a1", "a3"]


async def test_write_validation_errors_are_tool_errors(server):
    async with Client(server) as client:
        for args in (
            {"field": "x"},
            {"field": "x", "key": "k", "phase": 0.1},
            {"field": "x", "key": "k", "coherence": 2.0},
            {"field": "x", "key": "k" * 600},
            {"field": "x", "key": "k", "payload": "p" * 70_000},
        ):
            res = await client.call_tool("cdt_write", args)
            assert res.is_error, args


async def test_missing_field_is_tool_error_not_crash(server):
    async with Client(server) as client:
        res = await client.call_tool("cdt_consensus", {"field": "ghost"})
        assert res.is_error
        assert "ghost" in res.content[0].text
        res = await client.call_tool("cdt_write", {"field": "ghost", "key": "k", "auto_create": False})
        assert res.is_error
        # Server still healthy afterwards
        assert (await call(client, "cdt_list"))["fields"] == []


async def test_invalid_field_name_rejected(server):
    async with Client(server) as client:
        for bad in ("../etc/passwd", "", "-leading", "has space", "x" * 200):
            res = await client.call_tool("cdt_create", {"name": bad})
            assert res.is_error, bad
        res = await client.call_tool("cdt_create", {"name": "ok.name-1:v2@x"})
        assert not res.is_error
        res = await client.call_tool("cdt_create", {"name": "big", "bins": 10_000})
        assert res.is_error


async def test_create_is_idempotent(server):
    async with Client(server) as client:
        a = await call(client, "cdt_create", name="f", bins=16, decay_rate=0.5)
        b = await call(client, "cdt_create", name="f", bins=32)
        assert a["created"] is True and b["created"] is False
        assert b["field"]["bins"] == 16 and b["field"]["decay_rate"] == 0.5


async def test_snapshot_sync_roundtrip_is_idempotent(store, clock):
    remote_store = FieldStore(clock=clock)
    async with Client(create_server(store)) as local, Client(create_server(remote_store)) as remote:
        await call(remote, "cdt_write", field="shared", key="a", coherence=0.5, payload="A", agent_id="r1")
        await call(remote, "cdt_write", field="shared", key="b", coherence=0.9, payload="B", agent_id="r2")
        snap = (await call(remote, "cdt_snapshot", field="shared"))["snapshot"]
        assert "field" not in snap
        assert "field" in (await call(remote, "cdt_snapshot", field="shared", include_field=True))["snapshot"]

        await call(local, "cdt_write", field="shared", key="a", coherence=0.6, payload="A", agent_id="l1")
        s1 = await call(local, "cdt_sync", field="shared", snapshot=snap)
        assert s1["absorbed"] == 2 and s1["records"] == 3
        assert s1["consensus"]["top_payload"] == "A"  # 0.5 + 0.6 > 0.9

        s2 = await call(local, "cdt_sync", field="shared", snapshot=snap)
        assert s2["absorbed"] == 0 and s2["records"] == 3

        # Sync back the other way converges both replicas.
        local_snap = (await call(local, "cdt_snapshot", field="shared"))["snapshot"]
        s3 = await call(remote, "cdt_sync", field="shared", snapshot=local_snap)
        assert s3["absorbed"] == 1
        lc = await call(local, "cdt_consensus", field="shared")
        rc = await call(remote, "cdt_consensus", field="shared")
        assert lc["amplitude"] == pytest.approx(rc["amplitude"])
        assert lc["top_payload"] == rc["top_payload"]


async def test_sync_creates_field_when_missing_and_validates(server):
    async with Client(server) as client:
        res = await client.call_tool("cdt_sync", {"field": "new", "snapshot": {"records": "nope"}})
        assert res.is_error
        snap = {
            "name": "whatever",
            "bins": 8,
            "records": [{"id": "r1", "phase": 0.0, "value": 1.0, "coherence": 1.0, "timestamp": 1.0, "payload": "P"}],
        }
        s = await call(client, "cdt_sync", field="new", snapshot=snap)
        assert s["absorbed"] == 1
        assert (await call(client, "cdt_consensus", field="new"))["top_payload"] == "P"
        res = await client.call_tool("cdt_sync", {"field": "other", "snapshot": snap, "create": False})
        assert res.is_error
        # Bin mismatch on an existing field is rejected cleanly
        await call(client, "cdt_create", name="wide", bins=64)
        res = await client.call_tool("cdt_sync", {"field": "wide", "snapshot": snap})
        assert res.is_error and "bins" in res.content[0].text


async def test_merge_tool_superposes_sources(server):
    async with Client(server) as client:
        await call(client, "cdt_write", field="a", key="x", coherence=0.4, payload="X")
        await call(client, "cdt_write", field="b", key="x", coherence=0.4, payload="X")
        await call(client, "cdt_write", field="b", key="y", coherence=0.7, payload="Y")
        m = await call(client, "cdt_merge", sources=["a", "b"], into="ab")
        assert m["field"]["records"] == 3
        assert m["consensus"]["top_payload"] == "X"
        # Merging again into the existing target is idempotent
        m2 = await call(client, "cdt_merge", sources=["a", "b"], into="ab")
        assert m2["field"]["records"] == 3
        lst = await call(client, "cdt_list")
        by_name = {f["name"]: f for f in lst["fields"]}
        assert by_name["a"]["records"] == 1 and by_name["b"]["records"] == 2
        res = await client.call_tool("cdt_merge", {"sources": [], "into": "z"})
        assert res.is_error
        res = await client.call_tool("cdt_merge", {"sources": ["missing"], "into": "z"})
        assert res.is_error


async def test_decay_and_delete(server):
    async with Client(server) as client:
        await call(client, "cdt_write", field="f", key="k", coherence=1.0)
        d = await call(client, "cdt_decay", field="f", dt=1.0, rate=math.log(2))
        assert d["affected"] == 1 and d["spectral_density"] == pytest.approx(0.25)
        assert d["decay_id"] and d["factor"] == pytest.approx(0.5)
        snap = (await call(client, "cdt_snapshot", field="f"))["snapshot"]
        assert len(snap["decays"]) == 1
        d = await call(client, "cdt_decay", field="f", dt=100.0, rate=1.0)
        assert d["pruned"] == 1
        res = await client.call_tool("cdt_decay", {"field": "f", "dt": -1})
        assert res.is_error
        r = await call(client, "cdt_delete", field="f")
        assert r["deleted"] is True
        r = await call(client, "cdt_delete", field="f")
        assert r["deleted"] is False


async def test_resources_and_prompt(server):
    async with Client(server) as client:
        await call(client, "cdt_write", field="res", key="k", coherence=0.3, payload="K")
        body = json.loads((await client.read_resource("cdt://fields")).contents[0].text)
        assert body["fields"][0]["name"] == "res"
        snap = json.loads((await client.read_resource("cdt://field/res")).contents[0].text)
        assert len(snap["field"]) == 64 and snap["records"][0]["payload"] == "K"
        cons = json.loads((await client.read_resource("cdt://field/res/consensus")).contents[0].text)
        assert cons["top_payload"] == "K"
        prompt = await client.get_prompt("cdt_reconcile", {"field": "res", "topic": "naming"})
        assert "cdt_consensus" in prompt.messages[0].content.text


async def test_missing_resource_errors(server):
    async with Client(server) as client:
        with pytest.raises(MCPError):
            await client.read_resource("cdt://field/nope/consensus")


async def test_phase_of_matches_write(server):
    async with Client(server) as client:
        p = (await call(client, "cdt_phase_of", key="hello"))["phase"]
        w = await call(client, "cdt_write", field="p", key="hello")
        assert w["phase"] == pytest.approx(p)


async def test_persistence_survives_restart(tmp_path, clock):
    store = FieldStore(tmp_path, clock=clock)
    async with Client(create_server(store)) as c:
        await call(c, "cdt_write", field="durable", key="k", coherence=0.8, payload="kept", agent_id="x")
    assert (tmp_path / "durable.json").exists()
    assert not list(tmp_path.glob("*.tmp"))

    reloaded = FieldStore(tmp_path, clock=clock)
    async with Client(create_server(reloaded)) as c:
        cons = await call(c, "cdt_consensus", field="durable")
        assert cons["top_payload"] == "kept" and cons["amplitude"] == pytest.approx(0.8)
        await call(c, "cdt_delete", field="durable")
    assert not (tmp_path / "durable.json").exists()


async def test_concurrent_writes_all_land(server):
    import asyncio

    async with Client(server) as client:
        await asyncio.gather(
            *(call(client, "cdt_write", field="c", key=f"k{i % 4}", coherence=0.5, agent_id=f"a{i}") for i in range(40))
        )
        cons = await call(client, "cdt_consensus", field="c", top_k=4)
        assert cons["record_count"] == 40
        assert all(b["amplitude"] == pytest.approx(5.0) for b in cons["alternatives"])


async def test_decay_syncs_between_replicas(store, clock):
    remote_store = FieldStore(clock=clock)
    async with Client(create_server(store)) as local, Client(create_server(remote_store)) as remote:
        await call(local, "cdt_write", field="s", key="k", coherence=1.0)
        snap = (await call(local, "cdt_snapshot", field="s"))["snapshot"]
        await call(remote, "cdt_sync", field="s", snapshot=snap)
        await call(local, "cdt_decay", field="s", dt=1.0, rate=math.log(2), prune_below=0.0)
        snap = (await call(local, "cdt_snapshot", field="s"))["snapshot"]
        s = await call(remote, "cdt_sync", field="s", snapshot=snap)
        assert s["absorbed"] == 1
        lc = await call(local, "cdt_consensus", field="s")
        rc = await call(remote, "cdt_consensus", field="s")
        assert lc["amplitude"] == pytest.approx(0.5) == pytest.approx(rc["amplitude"])


async def test_compact_keeps_density_and_drops_records(server):
    async with Client(server) as client:
        for c in (0.3, 0.5, 0.7):
            await call(client, "cdt_write", field="r", key="k", coherence=c, payload="p", agent_id="a")
        await call(client, "cdt_write", field="r", key="k", coherence=0.2, value=-1.0, agent_id="b")
        before = await call(client, "cdt_consensus", field="r")
        c = await call(client, "cdt_compact", field="r")
        assert c["removed"] == 2 and c["records"] == 2
        assert c["spectral_density"] == pytest.approx(before["spectral_density"])
        after = await call(client, "cdt_consensus", field="r")
        assert after["top_payload"] == before["top_payload"]
        assert after["contest_ratio"] == pytest.approx(before["contest_ratio"])
        assert (await call(client, "cdt_compact", field="r"))["removed"] == 0  # idempotent
        snap = (await call(client, "cdt_snapshot", field="r"))["snapshot"]
        assert snap["schema_version"] == 3 and any(rec["subsumes"] for rec in snap["records"])
