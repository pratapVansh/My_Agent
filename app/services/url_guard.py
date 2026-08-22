"""
Outbound URL safety checks (SSRF guard).

Any server-side fetch driven by a user-supplied URL — the ERP attendance
scraper being the one in this codebase — can otherwise be pointed at internal
services, link-local cloud metadata endpoints, or loopback, turning the server
into a proxy for reaching things the caller cannot reach directly.

This module answers one question: is it safe for *this server* to fetch that
URL? It does not answer whether the caller is authorised to ask — that is an
authentication concern (see docs/AUDIT_REPORT.md, C1).
"""
from __future__ import annotations

import ipaddress
import logging
import socket
from typing import List, Tuple
from urllib.parse import urlparse

from app.config import settings

logger = logging.getLogger(__name__)

_ALLOWED_SCHEMES = frozenset({"http", "https"})

# Hostnames that resolve to infrastructure regardless of DNS answer.
_BLOCKED_HOSTNAMES = frozenset({
    "localhost",
    "metadata.google.internal",
    "metadata.goog",
})


class UnsafeURLError(ValueError):
    """Raised when a URL must not be fetched by the server."""


def _resolve_addresses(hostname: str) -> List[ipaddress._BaseAddress]:
    """Resolve a hostname to every IP it maps to (v4 and v6)."""
    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror as exc:
        raise UnsafeURLError(f"Hostname '{hostname}' could not be resolved.") from exc

    addresses = []
    for info in infos:
        sockaddr = info[4]
        try:
            addresses.append(ipaddress.ip_address(sockaddr[0]))
        except ValueError:
            continue
    if not addresses:
        raise UnsafeURLError(f"Hostname '{hostname}' resolved to no usable address.")
    return addresses


def _is_disallowed(address: ipaddress._BaseAddress) -> bool:
    """True for any address that is not a routable public destination."""
    return (
        address.is_private
        or address.is_loopback
        or address.is_link_local        # covers 169.254.169.254 metadata
        or address.is_reserved
        or address.is_multicast
        or address.is_unspecified
    )


def assert_safe_outbound_url(raw_url: str) -> Tuple[str, str]:
    """
    Validate that `raw_url` is safe for the server to fetch.

    Returns (normalized_url, hostname). Raises UnsafeURLError when the URL uses
    a non-HTTP scheme, is malformed, or resolves to a non-public address.

    Set ALLOW_PRIVATE_NETWORK_SCRAPING=true to permit private ranges when the
    target ERP genuinely lives on an internal network.
    """
    if not raw_url or not raw_url.strip():
        raise UnsafeURLError("URL is required.")

    parsed = urlparse(raw_url.strip())

    if parsed.scheme.lower() not in _ALLOWED_SCHEMES:
        raise UnsafeURLError("Only http:// and https:// URLs are permitted.")

    hostname = (parsed.hostname or "").strip().lower()
    if not hostname:
        raise UnsafeURLError("URL is missing a hostname.")

    if hostname in _BLOCKED_HOSTNAMES:
        raise UnsafeURLError(f"Host '{hostname}' is not an allowed destination.")

    if settings.allow_private_network_scraping:
        logger.warning(
            "SSRF guard bypassed for host=%s (ALLOW_PRIVATE_NETWORK_SCRAPING is on)",
            hostname,
        )
        return parsed.geturl(), hostname

    # A literal IP is checked directly; a name is checked against every address
    # it resolves to, so a DNS entry pointing at 127.0.0.1 is still rejected.
    try:
        addresses = [ipaddress.ip_address(hostname)]
    except ValueError:
        addresses = _resolve_addresses(hostname)

    for address in addresses:
        if _is_disallowed(address):
            logger.warning(
                "Blocked outbound fetch to host=%s (resolved to non-public %s)",
                hostname, address,
            )
            raise UnsafeURLError(
                f"Host '{hostname}' resolves to a non-public address and cannot be fetched."
            )

    return parsed.geturl(), hostname
