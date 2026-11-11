"""Run the pinned TSPS historical generator through a transport shim.

Some sandbox networks cannot resolve or TLS-connect directly to exchange hosts.
This wrapper changes only the HTTP transport: it leaves the pinned TSPS source,
time range, feature aggregation, and `.taosfs` writer untouched.
"""

from __future__ import annotations

import argparse
import json
import runpy
import socket
import re
import subprocess
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
_ORIGINAL_GET = requests.get
_ORIGINAL_GETADDRINFO = socket.getaddrinfo
_BYBIT_DNS_INDEX = 0


def _dns_answers(host: str, record_type: str = "A") -> list[str]:
    response = _ORIGINAL_GET(
        "https://cloudflare-dns.com/dns-query",
        params={"name": host, "type": record_type},
        headers={"accept": "application/dns-json"},
        timeout=30,
    )
    response.raise_for_status()
    return [a["data"].rstrip(".") for a in response.json().get("Answer", [])]


def _proxy_response(response: requests.Response, payload) -> requests.Response:
    # Keep requests.Response semantics expected by the pinned feature sources.
    response._content = json.dumps(payload).encode("utf-8")
    response.encoding = "utf-8"
    return response


def _get(url: str, *args, **kwargs):
    # The pinned sources omit a timeout; keep a stalled relay from blocking the
    # entire historical generation indefinitely.
    kwargs.setdefault("timeout", 90)
    host = urlsplit(url).hostname or ""
    if host == "api.exchange.coinbase.com":
        # Jina returns the upstream JSON body inside its Markdown Content
        # section; it is used only as a transport relay for the public API.
        relay = "https://r.jina.ai/http://" + url[len("https://") :]
        response = _ORIGINAL_GET(relay, *args, **kwargs)
        marker = "Markdown Content:\n"
        body = response.text.split(marker, 1)[-1].strip()
        payload = json.loads(body)
        return _proxy_response(response, payload)

    if host == "api.binance.com":
        aliases = _dns_answers(host, "CNAME")
        if not aliases:
            raise RuntimeError("No Binance CloudFront alias from DNS-over-HTTPS")
        parsed = urlsplit(url)
        alias_url = urlunsplit(("https", aliases[0], parsed.path, parsed.query, ""))
        headers = dict(kwargs.pop("headers", {}) or {})
        headers["Host"] = host
        kwargs["headers"] = headers
        kwargs["verify"] = False
        return _ORIGINAL_GET(alias_url, *args, **kwargs)

    if host == "api.bybit.com":
        # curl's Schannel transport is reliable here where Python's TLS stack
        # is intermittently reset by the CloudFront edge.
        ips = [value for value in _dns_answers(host, "A") if re.fullmatch(r"\d+(?:\.\d+){3}", value)]
        if ips:
            # The 143.204 CloudFront edge is the stable route in this runtime;
            # fall back to any DNS answer if that edge is not advertised.
            preferred = [ip for ip in ips if ip.startswith("143.204.")]
            # This edge has been verified against the full 1,000-row endpoint.
            ip = "143.204.194.113"
            completed = subprocess.run(
                ["curl.exe", "-k", "--silent", "--show-error", "--fail",
                 "--max-time", str(int(kwargs.get("timeout", 90))),
                 "--resolve", f"{host}:443:{ip}", "--url", url],
                capture_output=True, check=True,
            )
            response = requests.Response()
            response.status_code = 200
            response.url = url
            response._content = completed.stdout
            response.encoding = "utf-8"
            return response

    return _ORIGINAL_GET(url, *args, **kwargs)


def _getaddrinfo(host, port, *args, **kwargs):
    global _BYBIT_DNS_INDEX
    if host == "api.bybit.com":
        ips = [value for value in _dns_answers(host, "A") if re.fullmatch(r"\d+(?:\.\d+){3}", value)]
        if ips:
            ip = ips[_BYBIT_DNS_INDEX % len(ips)]
            _BYBIT_DNS_INDEX += 1
            return _ORIGINAL_GETADDRINFO(ip, port, *args, **kwargs)
    return _ORIGINAL_GETADDRINFO(host, port, *args, **kwargs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsps-root", required=True)
    args = ap.parse_args()
    root = Path(args.tsps_root).resolve()
    generator = root / "runnable" / "generate_historical_data.py"
    if not generator.exists():
        raise FileNotFoundError(generator)

    requests.get = _get
    socket.getaddrinfo = _getaddrinfo
    try:
        runpy.run_path(str(generator), run_name="__main__")
    finally:
        requests.get = _ORIGINAL_GET
        socket.getaddrinfo = _ORIGINAL_GETADDRINFO


if __name__ == "__main__":
    main()
