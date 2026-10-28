"""
Authentication — launch real Chrome, attach Playwright over CDP, capture session.

Why not launch the browser through Playwright? DeepSeek's sign-in routes through
Google, and Google's anti-automation fingerprint rejects any browser process that
was started by Playwright/Puppeteer (or advertises `navigator.webdriver`, or is the
bundled Chromium), regardless of launch flags — it shows "Couldn't sign you in.
This browser or app may not be secure."

So we do the opposite: start a NORMAL Chrome as an ordinary subprocess with only
`--remote-debugging-port` and a dedicated `--user-data-dir`, then attach Playwright
to the already-running browser via the Chrome DevTools Protocol (`connect_over_cdp`).
The browser was not launched by automation, so Google is not presented with any of
the signals it blocks on. The CDP connection itself is not a detectable signal —
every Chrome DevTools window uses it.

We then capture the bearer token from `localStorage.userToken` plus the session
cookies and hand them to the pure-HTTP client in `deepseek.client`.

    from deepseek.auth import get_session
    session = get_session()          # launches Chrome over CDP if no session yet
    print(len(session.token), "chars")
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import socket
import subprocess
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent

# The harness drives a dedicated Chrome profile (NOT the user's real Google-Chrome
# profile and NOT a Playwright-managed one). A dedicated --user-data-dir is required
# because Chrome 136+ ignores --remote-debugging-port when it points at the default
# profile; it works on a dedicated dir across all versions.
# Override with DEEPSEEK_CDP_PROFILE.
DEFAULT_CDP_PROFILE_DIR = Path(
    os.getenv("DEEPSEEK_CDP_PROFILE", ROOT / "session" / "chrome-cdp-profile")
)
# Backwards-compatible alias (kept so existing callers/imports keep working).
DEFAULT_PROFILE_DIR = DEFAULT_CDP_PROFILE_DIR
DEFAULT_SESSION_FILE = ROOT / "session" / "session.json"

CHAT_URL = "https://chat.deepseek.com/"
SIGNIN_URL = "https://chat.deepseek.com/sign_in"

# Chrome's remote-debugging port. A free port is chosen from this base at runtime.
_CDP_PORT_BASE = 9222

# Token is trusted for this long before we refresh it from the browser again.
SESSION_MAX_AGE = 6 * 60 * 60  # 6 hours

# Dedicated logger that always emits to stderr (message only), so the [auth] lines
# are visible both from the CLI (`python -m deepseek.auth`) and when the server
# calls this from a worker thread, without depending on uvicorn's config.
log = logging.getLogger("deepseek.auth")
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(_h)
    log.setLevel(logging.INFO)
    log.propagate = False


class LoginRequired(RuntimeError):
    """Raised when no usable session exists and interactive login is disallowed
    (e.g. inside the server, where we can't pop open a browser mid-request).
    The message tells the user how to log in."""

    DEFAULT = (
        "No DeepSeek session found. Log in first by running:\n"
        "    python -m deepseek.auth\n"
        "This opens Chrome once so you can sign in; afterwards the harness reuses\n"
        "the saved session (and the persistent CDP profile) automatically."
    )

    def __init__(self, message: str = DEFAULT):
        super().__init__(message)


@dataclass
class Session:
    """A captured signed-in DeepSeek session."""

    token: str
    cookies: Dict[str, str]
    user_agent: str
    captured_at: float

    @property
    def age(self) -> float:
        return time.time() - self.captured_at

    def save(self, path: Path = DEFAULT_SESSION_FILE) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path = DEFAULT_SESSION_FILE) -> Optional["Session"]:
        if not path.exists():
            return None
        try:
            return cls(**json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            return None


# --- in-page helpers --------------------------------------------------------

# Reads the bearer token the web app stores after login. Shape:
#   localStorage.userToken = {"value":"<TOKEN>","__version":"0"}
_READ_TOKEN_JS = """
() => {
  try {
    const raw = window.localStorage.getItem('userToken');
    if (!raw) return null;
    const o = JSON.parse(raw);
    return (o && o.value) ? o.value : null;
  } catch (e) { return null; }
}
"""


def _safe_evaluate(page, js: str):
    """Run page.evaluate, swallowing the benign 'Execution context was destroyed'
    error that fires when a navigation (e.g. the post-login redirect) happens to
    land mid-evaluate. Returns None on any such transient failure instead of
    raising, so callers can just retry on the next poll."""
    try:
        return page.evaluate(js)
    except Exception as e:
        msg = str(e)
        if "Execution context was destroyed" in msg or "navigation" in msg.lower():
            return None
        raise


def _capture_from_context(context, page) -> Optional[Session]:
    """Read token + cookies + UA off a logged-in page, or None if not signed in."""
    token = _safe_evaluate(page, _READ_TOKEN_JS)
    if not token:
        return None
    cookies = {c["name"]: c["value"] for c in context.cookies()}
    ua = _safe_evaluate(page, "() => navigator.userAgent") or ""
    return Session(token=token, cookies=cookies, user_agent=ua, captured_at=time.time())


def _wait_for_token(page, timeout: float) -> Optional[str]:
    """Poll localStorage.userToken until it appears or we time out. Tolerates
    transient navigations (e.g. the Google OAuth redirect chain) that briefly
    destroy the page's JS execution context — those just count as "no token
    yet" rather than aborting the whole login."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        token = _safe_evaluate(page, _READ_TOKEN_JS)
        if token:
            return token
        time.sleep(1)
    return None


def _safe_goto(page, url: str) -> None:
    """Navigate, tolerating the benign `net::ERR_ABORTED` that DeepSeek's SPA
    redirects and the AWS WAF check often raise mid-navigation. We wait only for
    the initial commit, not full `load`; the token-poll afterwards is what
    actually gates sign-in, so an aborted/partial load here is fine."""
    try:
        page.goto(url, wait_until="commit", timeout=60000)
    except Exception as e:
        log.warning("[auth] navigation to %s failed: %s: %s",
                    url, type(e).__name__, str(e).splitlines()[0] if str(e) else "")
    # Give the SPA a moment to render its UI before we touch the page.
    time.sleep(2)


# --- CDP browser launch/attach ---------------------------------------------

def _find_chrome_binary() -> Optional[str]:
    """Return the path to a system Chrome binary, or None."""
    for name in ("google-chrome-stable", "google-chrome",
                 "chromium-browser", "chromium"):
        path = shutil.which(name)
        if path:
            return path
    return None


def _free_cdp_port(start: int = _CDP_PORT_BASE) -> int:
    """Pick a free TCP port for Chrome's remote-debugging endpoint, starting at
    `start`. Chrome binds to 127.0.0.1, so binding the probe locally is enough."""
    for port in range(start, start + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No free CDP port in range {start}..{start + 49}")


def _launch_chrome_with_cdp(
    profile_dir: Path,
    url: str,
    port: int = _CDP_PORT_BASE,
    headless: bool = False,
) -> subprocess.Popen:
    """Launch a real Chrome as a normal subprocess with remote debugging enabled.

    The browser is NOT launched by Playwright; Playwright attaches to it afterwards
    via `connect_over_cdp`. No automation-hiding flags are passed on purpose — the
    process is an ordinary Chrome, which is exactly why Google does not block it.
    """
    chrome = _find_chrome_binary()
    if chrome is None:
        raise RuntimeError(
            "No system Chrome binary found (looked for google-chrome-stable, "
            "google-chrome, chromium-browser, chromium). Install Chrome or run "
            "`playwright install chromium` and re-point the harness.")
    profile_dir.mkdir(parents=True, exist_ok=True)
    args = [
        chrome,
        f"--remote-debugging-port={port}",
        f"--user-data-dir={profile_dir}",
        "--no-first-run",
        "--no-default-browser-check",
        url,
    ]
    if headless:
        args.insert(1, "--headless=new")
    return subprocess.Popen(
        args,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _wait_for_cdp(port: int, timeout: float = 30.0) -> str:
    """Poll http://localhost:<port>/json/version until Chrome is ready. Return the
    webSocketDebuggerUrl, or raise on timeout."""
    import urllib.request
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(
                f"http://localhost:{port}/json/version", timeout=2
            ) as resp:
                data = json.loads(resp.read())
                return data["webSocketDebuggerUrl"]
        except Exception:
            time.sleep(0.5)
    raise RuntimeError(f"Chrome CDP endpoint on port {port} did not become ready")


def _terminate(proc: subprocess.Popen) -> None:
    """Stop a Chrome we launched, gracefully then hard-kill."""
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _connect(p, port: int):
    """Attach Playwright to the running Chrome via CDP and hand back
    (browser, context, page). Uses the default profile context (index 0)."""
    browser = p.chromium.connect_over_cdp(f"http://localhost:{port}")
    if browser.contexts:
        context = browser.contexts[0]
    else:
        context = browser.new_context()
    page = context.pages[0] if context.pages else context.new_page()
    return browser, context, page


def login(
    profile_dir: Path = DEFAULT_PROFILE_DIR,
    headless: bool = False,
    assume_logged_out: bool = False,
) -> Session:
    """Open a real Chrome (via CDP) at DeepSeek and wait for a signed-in session.

    Chrome is started as a plain subprocess and Playwright attaches over CDP, so the
    browser is not an automation stack and Google's sign-in does not reject it. Once
    `localStorage.userToken` appears we capture token + cookies + UA, save the
    session, and close Chrome.

    `assume_logged_out=True` opens the sign-in URL directly (used when the caller
    has just confirmed there is no session). Otherwise we open the chat URL first so
    an already-signed-in dedicated profile is reused with no prompt.
    """
    profile_dir.mkdir(parents=True, exist_ok=True)
    url = SIGNIN_URL if assume_logged_out else CHAT_URL
    port = _free_cdp_port()
    proc: Optional[subprocess.Popen] = None
    try:
        proc = _launch_chrome_with_cdp(profile_dir, url, port=port, headless=headless)
        _wait_for_cdp(port)
        with sync_playwright() as p:
            browser, context, page = _connect(p, port)
            print("[auth] Chrome opened. Sign in to DeepSeek in the window.")
            print("[auth] Waiting for the session token...")
            token = _wait_for_token(page, timeout=300)
            if not token:
                raise RuntimeError("Login timed out — no token captured.")
            session = _capture_from_context(context, page)
            browser.close()
        if session is None:
            raise RuntimeError("Logged in but could not read the token.")
        session.save()
        return session
    finally:
        if proc is not None:
            _terminate(proc)


def _headless_refresh(profile_dir: Path = DEFAULT_PROFILE_DIR) -> Optional[Session]:
    """Try to capture a token by attaching to a headless Chrome on the dedicated
    profile. Returns a saved Session if that profile is still signed in, else None.
    Never opens a visible window.

    If the dedicated profile has no session (or Chrome cannot start), this returns
    None rather than raising — callers decide whether to fall back to an interactive
    login. Transient CDP/navigation failures are logged, not propagated."""
    profile_dir.mkdir(parents=True, exist_ok=True)
    port = _free_cdp_port()
    proc: Optional[subprocess.Popen] = None
    session: Optional[Session] = None
    try:
        proc = _launch_chrome_with_cdp(profile_dir, CHAT_URL, port=port, headless=True)
        _wait_for_cdp(port)
        with sync_playwright() as p:
            browser, context, page = _connect(p, port)
            _safe_goto(page, CHAT_URL)
            session = _capture_from_context(context, page)
            browser.close()
    except Exception as exc:
        log.warning("[auth] headless CDP refresh failed: %s: %s",
                    type(exc).__name__, str(exc).splitlines()[0] if str(exc) else "")
        session = None
    finally:
        if proc is not None:
            _terminate(proc)

    if session is not None:
        session.save()
    return session


def get_session(
    profile_dir: Path = DEFAULT_PROFILE_DIR,
    session_file: Path = DEFAULT_SESSION_FILE,
    max_age: int = SESSION_MAX_AGE,
    allow_interactive: bool = True,
) -> Session:
    """Return a usable session: cached file if fresh, else a headless CDP capture
    from the dedicated Chrome profile.

    If neither works and `allow_interactive` is True, launch Chrome for the user to
    sign in. If it's False (the server's case — we can't pop a browser mid-request),
    raise `LoginRequired`.

    Note: this launches Chrome and attaches Playwright's *sync* CDP API, so it must
    not be called from inside an asyncio event loop — call it from a worker thread
    (e.g. run_in_threadpool)."""
    cached = Session.load(session_file)
    if cached and cached.age < max_age:
        return cached

    # Try a headless capture from the (presumably signed-in) dedicated profile.
    session = _headless_refresh(profile_dir)
    if session is not None:
        return session

    if not allow_interactive:
        raise LoginRequired()

    # No session yet — launch Chrome so the user can sign in by hand.
    print("[auth] No valid session found — launching Chrome to sign in...")
    return login(profile_dir=profile_dir, assume_logged_out=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m deepseek.auth",
        description=(
            "Sign in to DeepSeek once and save the session. Launches your installed "
            "Chrome as a normal browser pointed at DeepSeek, attaches Playwright as a "
            "passive debugger over the Chrome DevTools Protocol, then captures the "
            "token from a dedicated profile at session/chrome-cdp-profile/. After you "
            "sign in once, later runs capture the token automatically."
        ),
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="run Chrome headless (only useful once the dedicated profile is signed in)",
    )
    args = parser.parse_args()

    try:
        s = login(headless=args.headless)
    except Exception as exc:
        print(f"[auth] login failed: {exc}")
        raise SystemExit(1)
    # Never print the token itself — only its length and the cookie count.
    print(f"[auth] captured a session token ({len(s.token)} chars, "
          f"{len(s.cookies)} cookies); saved to {DEFAULT_SESSION_FILE}")
