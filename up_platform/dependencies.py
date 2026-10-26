"""Load optional third-party libraries only when their capability is requested."""
from importlib import import_module, util


def dependency(name):
    try:
        return import_module(name)
    except (ImportError, OSError) as error:
        raise RuntimeError(f"Required platform dependency {name!r} is unavailable: {error}") from error
    except SystemExit as error:
        # Some optional SDKs call exit() when their own dependency is missing.
        # Keep that failure visible without terminating UP or hiding the cause.
        raise RuntimeError(f"Required platform dependency {name!r} exited during import "
                           f"(code={error.code!r}); check the SDK's preceding diagnostics") from error


def available(name):
    try:
        return util.find_spec(name) is not None
    except (ImportError, OSError, ValueError, SystemExit):
        return False

