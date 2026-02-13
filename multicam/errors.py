"""
Custom exceptions for handling failure/edge cases in the `multicam` package.

:copyright:
    2026, Conor A. Bacon.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""


class CaptureFailure(Exception):
    pass


class InvalidConfigFile(Exception):
    pass
