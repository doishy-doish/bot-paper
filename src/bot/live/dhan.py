#!/usr/bin/env python3
"""DhanHQ v2 REST client — CNC delivery only, AMO-first. No secrets in code.

Auth: 24h access token, generated per-run. Two modes:
  1. DHAN_ACCESS_TOKEN env (pasted daily from web.dhan.co) — simple, manual.
  2. Auto-generate in-job: needs DHAN_API_KEY + DHAN_API_SECRET (12-month) +
     DHAN_CLIENT_ID + DHAN_PIN + DHAN_TOTP_SECRET (TOTP enabled on account).
     Mode 2 is the automation path; all five live in CI secrets, never in git.

Docs: https://dhanhq.co/docs/v2/ (verified 2026-11-09: /orders, /super-order,
Forever Order, afterMarketOrder + amoTime PRE_OPEN/OPEN/OPEN_30/OPEN_60,
productType CNC/INTRADAY/MARGIN/MTF/CO/BO, API key valid 12 months.)
"""
import json
import os
import time
import urllib.request

BASE = os.environ.get("DHAN_BASE_URL", "https://api.dhan.co/v2")
AUTH = os.environ.get("DHAN_AUTH_URL", "https://auth.dhan.co/app")


class DhanError(RuntimeError):
    pass


def _req(method, url, token=None, app_id=None, app_secret=None, payload=None, timeout=20):
    data = json.dumps(payload).encode() if payload is not None else None
    r = urllib.request.Request(url, data=data, method=method,
                               headers={"Content-Type": "application/json"})
    if token:
        r.add_header("access-token", token)
    if app_id:
        r.add_header("app_id", app_id)
    if app_secret:
        r.add_header("app_secret", app_secret)
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            return json.loads(resp.read().decode() or "{}")
    except Exception as e:  # urllib HTTPError carries the broker's error body
        body = ""
        try:
            body = e.read().decode()[:500]
        except Exception:
            pass
        raise DhanError(f"{method} {url} failed: {e} {body}")


def generate_token_totp(api_key, api_secret, client_id, pin, totp_code):
    """Auto-generate a 24h access token (needs TOTP enabled on the account).

    totp_code: current 6-digit code. Callers in CI generate it from
    DHAN_TOTP_SECRET with pyotp at run time — the secret never leaves CI.
    """
    out = _req("POST",
               f"{AUTH}/generateAccessToken?dhanClientId={client_id}&pin={pin}&totp={totp_code}",
               app_id=api_key, app_secret=api_secret)
    token = out.get("accessToken") or out.get("token")
    if not token:
        raise DhanError(f"token generation returned no token: {str(out)[:300]}")
    return token


class DhanClient:
    def __init__(self, client_id, token, dry_run=True):
        self.cid = str(client_id)
        self.token = token
        self.dry_run = dry_run

    def _post(self, path, payload):
        if self.dry_run:
            return {"dry_run": True, "would_post": path, "payload": payload}
        return _req("POST", BASE + path, token=self.token, payload=payload)

    def _get(self, path):
        return _req("GET", BASE + path, token=self.token)

    def place_cnc_amo(self, side, security_id, qty, amo_time="OPEN",
                      limit_price=None, correlation_id=""):
        """One CNC delivery order for next-open execution via after-market route.

        side: BUY | SELL. MARKET when limit_price is None, else LIMIT.
        amo_time: OPEN (default) | PRE_OPEN | OPEN_30 | OPEN_60.
        productType is hard-pinned to CNC — anything else raises.
        """
        assert side in ("BUY", "SELL"), side
        assert amo_time in ("PRE_OPEN", "OPEN", "OPEN_30", "OPEN_60"), amo_time
        assert isinstance(qty, int) and qty > 0, qty
        payload = {
            "dhanClientId": self.cid,
            "correlationId": (correlation_id or f"q10-{int(time.time())}")[:30],
            "transactionType": side,
            "exchangeSegment": "NSE_EQ",
            "productType": "CNC",  # pinned: delivery only, never margin
            "orderType": "LIMIT" if limit_price else "MARKET",
            "validity": "DAY",
            "securityId": str(security_id),
            "quantity": qty,
            "afterMarketOrder": True,
            "amoTime": amo_time,
        }
        if limit_price:
            payload["price"] = float(limit_price)
        return self._post("/orders", payload)

    def order_status(self, order_id):
        return self._get(f"/orders/{order_id}")

    def funds(self):
        return self._get("/funds")

    def holdings(self):
        return self._get("/holdings")
