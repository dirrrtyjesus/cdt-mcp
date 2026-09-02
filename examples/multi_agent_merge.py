"""Three agents reconcile a merge strategy through one in-process CDT server.

Run:  python examples/multi_agent_merge.py

Swap `Client(create_server(...))` for `Client("http://localhost:8000/mcp")`
to talk to a running `cdt-mcp --transport streamable-http` instance.
"""

import asyncio
import json

from mcp.client import Client

from cdt_mcp.server import create_server
from cdt_mcp.store import FieldStore


async def main() -> None:
    async with Client(create_server(FieldStore())) as c:
        proposals = [
            dict(agent_id="agent-1", key="rebase", coherence=0.9, payload="Rebase onto main"),
            dict(agent_id="agent-2", key="merge-commit", coherence=0.7, payload="Merge commit"),
            dict(agent_id="agent-3", key="rebase", coherence=0.4, payload="Rebase onto main", value=-1.0),
        ]
        for p in proposals:
            r = await c.call_tool("cdt_write", {"field": "merge", **p})
            assert not r.is_error, r.content
            out = r.structured_content
            print(f"{p['agent_id']:8s} -> bin {out['bin']:2d} weight {out['weight']:+.2f}")

        r = await c.call_tool("cdt_consensus", {"field": "merge", "top_k": 2})
        out = r.structured_content
        print("\nconsensus:", out["top_payload"], f"(share {out['share']:.3f})")
        for alt in out["alternatives"]:
            for pw in alt["payloads"]:
                print(f"  {pw['payload']:20s} weight {pw['weight']:+.2f} agents {pw['agents']}")

        snap = (await c.call_tool("cdt_snapshot", {"field": "merge"})).structured_content["snapshot"]
        print("\nsnapshot bytes:", len(json.dumps(snap)))
        again = await c.call_tool("cdt_sync", {"field": "merge", "snapshot": snap})
        print("re-sync absorbed:", again.structured_content["absorbed"], "(idempotent)")


if __name__ == "__main__":
    asyncio.run(main())
