"""Runs predictions off the UI thread and delivers results back on it."""

import logging
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)
_POLL_MS = 15


class PredictionWorker:
    """Executes one prediction at a time on a background thread.

    Tk must only be touched from the UI thread, so the worker never calls
    back directly: the UI thread polls for the result with `after`.
    """

    def __init__(self, master, completer):
        self._master = master
        self._completer = completer
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._future = None
        self._poll_timer = None
        self._closed = False

    def request(self, context, on_result):
        """Complete for `context`; call `on_result(context, completion)` when done."""
        self.cancel()
        self._future = self._executor.submit(
            self._completer.complete, context.prompt, context.prefix
        )
        self._poll(self._future, context, on_result)

    def cancel(self):
        """Forget the current request; its result will not be delivered."""
        if self._poll_timer:
            self._master.after_cancel(self._poll_timer)
            self._poll_timer = None
        if self._future:
            self._future.cancel()
            self._future = None

    def shutdown(self):
        """Stop delivering results and release the thread."""
        self._closed = True
        self.cancel()
        self._executor.shutdown(wait=False)

    def _poll(self, future, context, on_result):
        self._poll_timer = None
        if self._closed or future.cancelled():
            return
        if not future.done():
            self._poll_timer = self._master.after(
                _POLL_MS, self._poll, future, context, on_result
            )
            return
        if future.exception():
            logger.error("Prediction failed", exc_info=future.exception())
            return
        on_result(context, future.result())
