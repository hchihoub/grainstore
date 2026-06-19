//! Crate error type. Library code never panics on recoverable conditions; it
//! returns [`Error`]. Panics are reserved for genuine invariant violations
//! (poisoned locks), which indicate a bug, not an operational error.

use std::fmt;

/// All failures surfaced by the truth store.
#[derive(Debug)]
pub enum Error {
    /// An I/O failure against the WAL file.
    Io(std::io::Error),
    /// A WAL frame or stored value failed structural/CRC validation.
    Corrupt(String),
    /// The background committer is gone (shutdown or a prior fatal I/O error).
    Closed,
    /// A governance policy denied the operation (ACL or budget).
    Denied(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "i/o error: {e}"),
            Error::Corrupt(m) => write!(f, "corrupt data: {m}"),
            Error::Closed => write!(f, "committer is closed"),
            Error::Denied(m) => write!(f, "denied: {m}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

/// Convenience alias used throughout the crate.
pub type Result<T> = std::result::Result<T, Error>;
