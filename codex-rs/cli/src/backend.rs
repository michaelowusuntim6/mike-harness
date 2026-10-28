//! Lifecycle management for the bundled DeepSeek Python backend.
//!
//! `mike` ships with its bridge (`backend/app.py`) so the CLI can start it on
//! demand. If port 8000 is already serving, the CLI attaches to the existing
//! process and leaves it running on exit; otherwise it spawns the backend as a
//! child and terminates it when the CLI exits.

use anyhow::Context as _;
use anyhow::Result;
use anyhow::bail;
use std::fs::OpenOptions;
use std::io::Read;
use std::io::Write;
use std::net::TcpStream;
use std::net::ToSocketAddrs;
use std::path::Path;
use std::path::PathBuf;
use std::process::Child;
use std::process::Command;
use std::process::Stdio;
use std::sync::Once;
use std::sync::atomic::AtomicI32;
use std::sync::atomic::Ordering;
use std::time::Duration;
use std::time::Instant;

#[cfg(unix)]
use std::os::unix::process::CommandExt as _;

const DEFAULT_HOST: &str = "127.0.0.1";
const DEFAULT_PORT: u16 = 8000;
const CONNECT_TIMEOUT: Duration = Duration::from_millis(400);
const STARTUP_TIMEOUT: Duration = Duration::from_secs(30);
const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(5);

/// PID of the backend this process spawned, or 0 when attached/none.
static BACKEND_PID: AtomicI32 = AtomicI32::new(0);
static ATEXIT_ONCE: Once = Once::new();

/// Owns a spawned backend. Dropping it terminates the child; a CLI that
/// attached to an already-running backend holds an empty guard and never
/// touches that process.
pub struct BackendGuard {
    child: Option<Child>,
}

impl BackendGuard {
    /// Ensures a backend is reachable on the configured host/port, spawning the
    /// bundled one when necessary.
    pub fn ensure() -> Result<Self> {
        if env_flag("MIKE_NO_BRIDGE") {
            return Ok(Self { child: None });
        }

        let host = std::env::var("MIKE_BRIDGE_HOST").unwrap_or_else(|_| DEFAULT_HOST.to_string());
        let port = std::env::var("MIKE_BRIDGE_PORT")
            .ok()
            .and_then(|value| value.parse::<u16>().ok())
            .unwrap_or(DEFAULT_PORT);

        if port_open(&host, port) {
            return Ok(Self { child: None });
        }

        let backend_dir = find_backend_dir()?;
        let python = find_python(&backend_dir);
        let log_path = log_path();
        if let Some(parent) = log_path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating log directory {}", parent.display()))?;
        }
        let log = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)
            .with_context(|| format!("opening backend log {}", log_path.display()))?;
        let log_err = log
            .try_clone()
            .with_context(|| format!("cloning backend log {}", log_path.display()))?;

        let mut command = Command::new(&python);
        command
            .arg("app.py")
            .current_dir(&backend_dir)
            .env("HOST", &host)
            .env("PORT", port.to_string())
            .stdin(Stdio::null())
            .stdout(Stdio::from(log))
            .stderr(Stdio::from(log_err));

        #[cfg(unix)]
        command.process_group(0);

        let mut child = command.spawn().with_context(|| {
            format!(
                "starting DeepSeek backend with {} in {}",
                python.display(),
                backend_dir.display()
            )
        })?;

        let pid = i32::try_from(child.id()).unwrap_or(0);
        BACKEND_PID.store(pid, Ordering::SeqCst);
        register_atexit_cleanup();

        if let Err(error) = wait_until_ready(&mut child, &host, port) {
            terminate(&mut child);
            BACKEND_PID.store(0, Ordering::SeqCst);
            let tail = log_tail(&log_path);
            return Err(error.context(format!(
                "DeepSeek backend did not become ready; see {}\n{tail}",
                log_path.display()
            )));
        }

        Ok(Self { child: Some(child) })
    }
}

impl Drop for BackendGuard {
    fn drop(&mut self) {
        if let Some(mut child) = self.child.take() {
            BACKEND_PID.store(0, Ordering::SeqCst);
            terminate(&mut child);
        }
    }
}

fn wait_until_ready(child: &mut Child, host: &str, port: u16) -> Result<()> {
    let deadline = Instant::now() + STARTUP_TIMEOUT;
    loop {
        if let Some(status) = child.try_wait().context("checking backend process")? {
            bail!("DeepSeek backend exited during startup ({status})");
        }
        if http_ready(host, port) {
            return Ok(());
        }
        if Instant::now() >= deadline {
            bail!(
                "timed out after {}s waiting for the DeepSeek backend",
                STARTUP_TIMEOUT.as_secs()
            );
        }
        std::thread::sleep(Duration::from_millis(150));
    }
}

fn terminate(child: &mut Child) {
    #[cfg(unix)]
    {
        let pid = i32::try_from(child.id()).unwrap_or(0);
        if pid > 0 {
            // The child is its own process group leader; signal the whole group
            // so helper processes exit with it.
            unsafe {
                libc::kill(-pid, libc::SIGTERM);
            }
        }
    }
    let deadline = Instant::now() + SHUTDOWN_TIMEOUT;
    loop {
        match child.try_wait() {
            Ok(Some(_)) => return,
            Ok(None) if Instant::now() < deadline => std::thread::sleep(Duration::from_millis(50)),
            _ => break,
        }
    }
    #[cfg(unix)]
    {
        let pid = i32::try_from(child.id()).unwrap_or(0);
        if pid > 0 {
            unsafe {
                libc::kill(-pid, libc::SIGKILL);
            }
        }
    }
    let _ = child.kill();
    let _ = child.wait();
}

fn port_open(host: &str, port: u16) -> bool {
    connect(host, port, CONNECT_TIMEOUT).is_ok()
}

fn http_ready(host: &str, port: u16) -> bool {
    let Ok(mut stream) = connect(host, port, CONNECT_TIMEOUT) else {
        return false;
    };
    let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
    let _ = stream.set_write_timeout(Some(Duration::from_secs(2)));
    let request = format!(
        "GET /v1/models HTTP/1.1\r\nHost: {host}:{port}\r\nConnection: close\r\n\r\n"
    );
    if stream.write_all(request.as_bytes()).is_err() {
        return false;
    }
    let mut response = Vec::new();
    let mut buffer = [0u8; 4096];
    loop {
        match stream.read(&mut buffer) {
            Ok(0) => break,
            Ok(n) => {
                response.extend_from_slice(&buffer[..n]);
                if response.len() > 64 * 1024 {
                    break;
                }
            }
            Err(error)
                if matches!(
                    error.kind(),
                    std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
                ) =>
            {
                break;
            }
            Err(_) => return false,
        }
    }
    let response = String::from_utf8_lossy(&response);
    response.starts_with("HTTP/") && response.contains(" 200")
}

fn connect(host: &str, port: u16, timeout: Duration) -> std::io::Result<TcpStream> {
    let address = (host, port)
        .to_socket_addrs()?
        .next()
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, "no address resolved"))?;
    TcpStream::connect_timeout(&address, timeout)
}

fn find_backend_dir() -> Result<PathBuf> {
    if let Some(dir) = std::env::var_os("MIKE_BACKEND_DIR")
        .map(PathBuf::from)
        .filter(|path| !path.as_os_str().is_empty())
    {
        return validate_backend_dir(dir);
    }

    let mut candidates = Vec::new();
    if let Ok(exe) = std::env::current_exe()
        && let Some(bin_dir) = exe.parent()
    {
        candidates.push(bin_dir.join("backend"));
        candidates.push(bin_dir.join("../lib/mike/backend"));
        candidates.push(bin_dir.join("../share/mike/backend"));
    }
    // Development builds: <repo>/codex-rs/cli/../../backend
    candidates.push(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../backend"));
    if let Some(home) = codex_home() {
        candidates.push(home.join("backend"));
    }

    for candidate in candidates {
        if candidate.join("app.py").is_file() {
            return Ok(candidate.canonicalize().unwrap_or(candidate));
        }
    }

    bail!(
        "bundled DeepSeek backend not found; set MIKE_BACKEND_DIR to the directory containing app.py"
    )
}

fn validate_backend_dir(dir: PathBuf) -> Result<PathBuf> {
    if dir.join("app.py").is_file() {
        Ok(dir.canonicalize().unwrap_or(dir))
    } else {
        bail!(
            "MIKE_BACKEND_DIR={} does not contain app.py",
            dir.display()
        )
    }
}

fn find_python(backend_dir: &Path) -> PathBuf {
    let venv_python = if cfg!(windows) {
        backend_dir.join("venv/Scripts/python.exe")
    } else {
        backend_dir.join("venv/bin/python")
    };
    if venv_python.is_file() {
        return venv_python;
    }
    PathBuf::from("python3")
}

fn codex_home() -> Option<PathBuf> {
    std::env::var_os("CODEX_HOME")
        .map(PathBuf::from)
        .filter(|path| !path.as_os_str().is_empty())
        .or_else(|| {
            std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".mike"))
        })
}

fn log_path() -> PathBuf {
    codex_home()
        .unwrap_or_else(std::env::temp_dir)
        .join("logs")
        .join("backend.log")
}

fn log_tail(path: &Path) -> String {
    let Ok(contents) = std::fs::read(path) else {
        return String::new();
    };
    let start = contents.len().saturating_sub(4000);
    String::from_utf8_lossy(&contents[start..]).into_owned()
}

fn env_flag(name: &str) -> bool {
    std::env::var(name)
        .map(|value| !matches!(value.as_str(), "" | "0" | "false" | "no" | "off"))
        .unwrap_or(false)
}

fn register_atexit_cleanup() {
    ATEXIT_ONCE.call_once(|| {
        #[cfg(unix)]
        unsafe {
            libc::atexit(terminate_at_exit);
        }
    });
}

#[cfg(unix)]
extern "C" fn terminate_at_exit() {
    let pid = BACKEND_PID.swap(0, Ordering::SeqCst);
    if pid > 0 {
        unsafe {
            libc::kill(-pid, libc::SIGTERM);
        }
    }
}
