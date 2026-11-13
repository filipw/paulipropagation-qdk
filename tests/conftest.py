"""A cap on the memory of a test run.

A propagation that fails to truncate or to merge its terms grows without end, and by the time
a test would notice, the machine has no memory left. The tests need a few hundred megabytes,
so a run that has held more than the cap is ended at once.
"""

import os
import sys
import threading
import time

MEMORY_CAP = 3 << 30


def _end_the_run_at_the_cap(config) -> None:
    import resource

    # The largest resident size so far, which macOS gives in bytes and Linux in kilobytes.
    unit = 1 if sys.platform == "darwin" else 1024
    while True:
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * unit
        if peak > MEMORY_CAP:
            # The message has to get past the capture of the output of the running test.
            capture = config.pluginmanager.getplugin("capturemanager")
            if capture is not None:
                capture.suspend_global_capture(in_=True)
            message = f"\nthe test run was ended: it held {peak >> 20} MB, and the cap is {MEMORY_CAP >> 20} MB\n"
            os.write(2, message.encode())
            os._exit(3)
        time.sleep(0.2)


def pytest_configure(config) -> None:
    try:
        import resource  # noqa: F401
    except ImportError:  # not on Windows
        return
    threading.Thread(target=_end_the_run_at_the_cap, args=(config,), daemon=True).start()
