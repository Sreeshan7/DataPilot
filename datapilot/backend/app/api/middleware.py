"""Request-size protection for upload endpoints.

Starlette writes a multipart upload to disk *before* the route sees it, so a limit enforced
inside the route would only run after an attacker had already filled the disk. This
middleware therefore counts bytes as they arrive and stops the request at the limit, and
rejects an honest-but-too-large ``Content-Length`` before reading anything.

Raising from ``receive`` happens inside the route, where FastAPI's exception handling turns an
``HTTPException`` into a normal JSON error; the early check answers directly.
"""

from __future__ import annotations

import json

from fastapi import HTTPException

from app.schemas.project import ErrorCode, ErrorDetail, ErrorResponse


def _too_large(limit: int) -> dict:
    return ErrorResponse(
        error=ErrorDetail(
            code=ErrorCode.FILE_TOO_LARGE,
            message=f"The request is larger than the {limit // (1024 * 1024)} MiB limit.",
            details={"max_bytes": limit},
        )
    ).model_dump()


class BodySizeLimitMiddleware:
    """Pure ASGI middleware limiting the body size of selected paths."""

    def __init__(self, app, limits: dict[str, int]) -> None:
        self.app = app
        self.limits = limits

    def _limit_for(self, path: str):
        for suffix, limit in self.limits.items():
            if path.endswith(suffix):
                return limit
        return None

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http" or scope.get("method") not in {"POST", "PUT", "PATCH"}:
            await self.app(scope, receive, send)
            return
        limit = self._limit_for(scope.get("path", ""))
        if limit is None:
            await self.app(scope, receive, send)
            return

        headers = {k.lower(): v for k, v in scope.get("headers", [])}
        declared = headers.get(b"content-length")
        if declared is not None and declared.isdigit() and int(declared) > limit:
            body = json.dumps({"detail": _too_large(limit)}).encode()
            await send({"type": "http.response.start", "status": 413,
                        "headers": [(b"content-type", b"application/json"),
                                    (b"content-length", str(len(body)).encode())]})
            await send({"type": "http.response.body", "body": body})
            return

        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    raise HTTPException(status_code=413, detail=_too_large(limit))
            return message

        await self.app(scope, limited_receive, send)
