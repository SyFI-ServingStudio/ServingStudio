import json

import torch

from flashinfer import fused_moe


_original = fused_moe.trtllm_fp4_block_scale_routed_moe
_seen = set()


def _probe(*args, **kwargs):
    routing = kwargs.get("topk_ids")
    if isinstance(routing, tuple):
        ids = routing[0]
    else:
        ids = routing
    if torch.is_tensor(ids):
        counts = torch.bincount(ids.detach().reshape(-1).to("cpu"), minlength=112)
        value = tuple(int(x) for x in counts.tolist())
        if value not in _seen:
            _seen.add(value)
            print(
                "ROUTING_HISTOGRAM "
                + json.dumps({"counts": value, "sum": sum(value)}),
                flush=True,
            )
    return _original(*args, **kwargs)


fused_moe.trtllm_fp4_block_scale_routed_moe = _probe
