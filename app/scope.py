"""The one place a request's scope becomes a store predicate.

Files and their chunks are owner-scoped and carry raw content, so scope is
enforced structurally rather than remembered by each caller. Every route that
reads or removes stored content builds its predicate from :class:`ScopeFilter`.
Drift between hand-written scope clauses is the realistic failure mode; one
builder makes it impossible rather than merely reviewable.

The predicate is always ``(owner, file)`` and always goes into the store query
*before* ranking. Nothing downstream re-derives authorization from what the
store returned.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from fastapi import Request

PUBLIC_OWNER = "public"

OWNER_METADATA_KEY = "user_id"


@dataclass(frozen=True)
class ScopeFilter:
    """The owners a request may read, as a store-query clause."""

    owners: Tuple[str, ...]

    def owner_clause(self) -> Dict[str, Any]:
        return {OWNER_METADATA_KEY: {"$in": list(self.owners)}}

    def predicate(self, *clauses: Dict[str, Any]) -> Dict[str, Any]:
        """Combine caller clauses (e.g. file id) with the owner clause."""
        return {"$and": [*clauses, self.owner_clause()]}

    def owns(self, owner: Optional[str]) -> bool:
        """Whether a stored row's recorded owner falls inside this scope.

        A row with no recorded owner is *not* in scope. Before this release an
        absent ``user_id`` read as "belongs to everyone", which made any such
        chunk readable by every caller.
        """
        return owner in self.owners


def file_clause(file_id: str) -> Dict[str, Any]:
    return {"file_id": {"$eq": file_id}}


def files_clause(file_ids: Sequence[str]) -> Dict[str, Any]:
    return {"file_id": {"$in": list(file_ids)}}


def resolve_scope(request: Request, entity_id: Optional[str] = None) -> ScopeFilter:
    """Resolve the owners this request may read.

    The caller's own identity always comes from the verified token — never from
    the query string, the body, or the rows the store returned.

    A caller-supplied ``entity_id`` *widens* the owner set rather than replacing
    it, which is the change that closes the read paths below. It remains
    caller-asserted: this release does not yet carry a claim that proves the
    caller may act for that entity, so a deployment that exposes this API to
    untrusted callers must still authorize entity access upstream. See
    ``README.md``.
    """
    user = getattr(request.state, "user", None)

    if user is None:
        # No signing key configured anywhere: preserve the unauthenticated
        # deployment's single-owner behaviour rather than opening the store.
        return ScopeFilter(owners=(entity_id or PUBLIC_OWNER,))

    owners = {user.get("id") or PUBLIC_OWNER}
    if entity_id:
        owners.add(entity_id)

    return ScopeFilter(owners=tuple(sorted(owners)))


def resolve_owner_scope(
    request: Request, entity_id: Optional[str] = None
) -> ScopeFilter:
    """Resolve the single owner a presence check asks about.

    :func:`resolve_scope` reads the union of the caller and the named entity,
    which is right for retrieval but cannot say *whose* chunks a file has: a copy
    the caller embedded under their own id answers for the entity too. A presence
    check needs one namespace, so a named entity replaces the caller here rather
    than widening the set. That only ever narrows what ``resolve_scope`` would
    read for the same request, so it reaches nothing the caller could not already
    query.
    """
    if entity_id:
        return ScopeFilter(owners=(entity_id,))
    return resolve_scope(request)


def scope_owners(scope: ScopeFilter) -> List[str]:
    return list(scope.owners)
