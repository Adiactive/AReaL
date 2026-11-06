"""Run Ubuntu's repository helper with an optional, process-local TLS bypass."""

import functools
import inspect
import os
import runpy
import sys


def disable_httplib2_verification(http_class):
    """Cover Launchpad's httplib2 subclasses without changing installed files."""
    original = http_class.__init__
    signature = inspect.signature(original)

    @functools.wraps(original)
    def initialize(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        bound.arguments["disable_ssl_certificate_validation"] = True
        original(*bound.args, **bound.kwargs)

    http_class.__init__ = initialize


def main():
    command = "/usr/bin/add-apt-repository"
    mode = os.environ.get("SWE_BUILD_INSECURE_TLS", "0")
    if mode == "0":
        os.execv(command, [command, *sys.argv[1:]])
    if mode != "1":
        raise SystemExit("SWE_BUILD_INSECURE_TLS must be 0 or 1")

    import httplib2

    disable_httplib2_verification(httplib2.Http)
    sys.argv[0] = command
    runpy.run_path(command, run_name="__main__")


if __name__ == "__main__":
    main()
