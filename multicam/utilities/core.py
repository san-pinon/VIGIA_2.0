"""

:copyright:
    2026, Conor A. Bacon.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

import subprocess


def rsync(
    source: str,
    destination: str,
    remove_source: bool = False,
    mkpath: bool = False,
) -> int:
    """
    Utility function that attempts an rsync command a number of times.

    Parameters
    ----------
    source:
        Location of file to be sync'd.
    destination:
        Location to which file is to be sync'd.
    remove_source:
        Toggle for whether the source file is removed.
    mkpath:
        Construct the destination's path component.

    Returns
    -------
    return_code:
        0 = success, anything else = failure.

    """

    command = ["rsync", "-avq"]
    if remove_source:
        command.append("--remove-source-files")
    if mkpath:
        command.append("--mkpath")
    command.extend([str(source), str(destination)])

    return subprocess.Popen(command).wait()
