import logging
import sys
import traceback
from datetime import datetime


def configure_logging(log_file):
    """
    Configure logging to output to both a file and the console with elapsed time formatting.

    Sets up logging handlers and a custom exception hook for uncaught exceptions.

    Parameters
    ----------
    log_file : str
        Path to the log file where logs will be written.

    Returns
    -------
    None

    Examples
    --------
    >>> configure_logging('experiment.log')
    """
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(ElapsedFormatter())

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(ElapsedFormatter())

    logging.basicConfig(
        level=logging.INFO, handlers=[file_handler, stream_handler], force=True
    )

    def handler(type, value, tb):
        logging.exception("Uncaught exception: %s", str(value))
        logging.exception("\n".join(traceback.format_exception(type, value, tb)))

    sys.excepthook = handler


class ElapsedFormatter:
    """
    Formatter for logging that prepends elapsed time since initialization.

    Parameters
    ----------
    None

    Attributes
    ----------
    start_time : datetime.datetime
        The time when the formatter was initialized.

    Examples
    --------
    >>> formatter = ElapsedFormatter()
    >>> formatter.format_time(datetime.now() - formatter.start_time)
    '0:00:01'
    """

    def __init__(self):
        """
        Initialize the ElapsedFormatter and record the start time.

        Returns
        -------
        None
        """
        self.start_time = datetime.now()

    def format_time(self, t):
        """
        Format a timedelta as a string, omitting microseconds.

        Parameters
        ----------
        t : datetime.timedelta
            Time difference to format.

        Returns
        -------
        str
            Formatted time string.
        """
        return str(t)[:-7]

    def format(self, record):
        """
        Format a log record with elapsed time and log level.

        Parameters
        ----------
        record : logging.LogRecord
            The log record to format.

        Returns
        -------
        str
            Formatted log string.
        """
        elapsed_time = self.format_time(datetime.now() - self.start_time)
        log_str = "[%s %s]: %s" % (elapsed_time, record.levelname, record.getMessage())
        return log_str
