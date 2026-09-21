"""
SSRF guard tests (audit finding C3).

The attendance scraper fetches a caller-supplied URL, so these cases are the
boundary between "scrape my college portal" and "use the server as a proxy to
read internal services".
"""
import pytest

from app.services.url_guard import UnsafeURLError, assert_safe_outbound_url


@pytest.mark.parametrize("url", [
    "http://127.0.0.1/admin",
    "http://localhost:8080/",
    "https://10.0.0.5/internal",
    "http://192.168.1.1/router",
    "http://172.16.0.10/service",
    "http://169.254.169.254/latest/meta-data/",   # cloud metadata endpoint
    "http://[::1]/",
    "http://0.0.0.0/",
])
def test_rejects_non_public_destinations(url):
    with pytest.raises(UnsafeURLError):
        assert_safe_outbound_url(url)


@pytest.mark.parametrize("url", [
    "file:///etc/passwd",
    "gopher://example.com/",
    "ftp://example.com/x",
    "javascript:alert(1)",
])
def test_rejects_non_http_schemes(url):
    with pytest.raises(UnsafeURLError):
        assert_safe_outbound_url(url)


@pytest.mark.parametrize("url", ["", "   ", "not-a-url", "https://"])
def test_rejects_malformed_urls(url):
    with pytest.raises(UnsafeURLError):
        assert_safe_outbound_url(url)


def test_allows_public_https_host():
    normalized, hostname = assert_safe_outbound_url("https://example.com/erp/login.php")
    assert hostname == "example.com"
    assert normalized.startswith("https://example.com")


def test_hostname_is_lowercased_for_comparison():
    _, hostname = assert_safe_outbound_url("https://EXAMPLE.com/Login")
    assert hostname == "example.com"
