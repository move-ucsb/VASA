# __init__ for osgeo package.

# unofficial Windows binaries: set GDAL environment variables if necessary
from sys import version_info
import os

try:
    _here = os.path.dirname(__file__)
    if _here not in os.environ['PATH']:
        os.environ['PATH'] = _here + ';' + os.environ['PATH']
    if 'GDAL_DATA' not in os.environ:
        os.environ['GDAL_DATA'] = os.path.join(_here, 'data', 'gdal')
    if 'PROJ_LIB' not in os.environ:
        os.environ['PROJ_LIB'] = os.path.join(_here, 'data', 'proj')
    if 'GDAL_DRIVER_PATH' not in os.environ:
        os.environ['GDAL_DRIVER_PATH'] = os.path.join(_here, 'gdalplugins')
    os.add_dll_directory(_here)
except Exception:
    pass

from . import _gdal

__version__ = _gdal.__version__ = _gdal.VersionInfo("RELEASE_NAME")

gdal_version = tuple(int(s) for s in str(__version__).split('.') if s.isdigit())[:3]
python_version = tuple(version_info)[:3]

# Setting this flag to True will cause importing osgeo to fail on an unsupported Python version.
# Otherwise a deprecation warning will be issued instead.
# Importing osgeo fom an unsupported Python version might still partially work
# because the core of GDAL Python bindings might still support an older Python version.
# Hence the default option to just issue a warning.
# To get complete functionality upgrading to the minimum supported version is needed.
fail_on_unsupported_version = False

# The following is a Sequence of tuples in the form of (gdal_version, python_version).
# Each line represents the minimum supported Python version of a given GDAL version.
# Introducing a new line for the next GDAL version will trigger a deprecation warning
# when importing osgeo from a Python version which will not be
# supported in the next version of GDAL.
gdal_version_and_min_supported_python_version = (
    ((3, 2), (2, 0)),
    ((3, 3), (3, 6)),
    # ((3, 4), (3, 7)),
    # ((3, 5), (3, 8)),
)


def ver_str(ver):
    return '.'.join(str(v) for v in ver) if ver is not None else None


minimum_supported_python_version_for_this_gdal_version = None
this_python_version_will_be_deprecated_in_gdal_version = None
last_gdal_version_to_supported_your_python_version = None
next_version_of_gdal_will_use_python_version = None
for gdal_ver, py_ver in gdal_version_and_min_supported_python_version:
    if gdal_version >= gdal_ver:
        minimum_supported_python_version_for_this_gdal_version = py_ver
    if python_version >= py_ver:
        last_gdal_version_to_supported_your_python_version = gdal_ver
    if not this_python_version_will_be_deprecated_in_gdal_version:
        if python_version < py_ver:
            this_python_version_will_be_deprecated_in_gdal_version = gdal_ver
            next_version_of_gdal_will_use_python_version = py_ver


if python_version < minimum_supported_python_version_for_this_gdal_version:
    msg = 'Your Python version is {}, which is no longer supported by GDAL {}. ' \
          'Please upgrade your Python version to Python >= {}, ' \
          'or use GDAL <= {}, which supports your Python version.'.\
        format(ver_str(python_version), ver_str(gdal_version),
               ver_str(minimum_supported_python_version_for_this_gdal_version),
               ver_str(last_gdal_version_to_supported_your_python_version))

    if fail_on_unsupported_version:
        raise Exception(msg)
    else:
        from warnings import warn, simplefilter
        simplefilter('always', DeprecationWarning)
        warn(msg, DeprecationWarning)
elif this_python_version_will_be_deprecated_in_gdal_version:
    msg = 'You are using Python {} with GDAL {}. ' \
          'This Python version will be deprecated in GDAL {}. ' \
          'Please consider upgrading your Python version to Python >= {}, ' \
          'Which will be the minimum supported Python version of GDAL {}.'.\
        format(ver_str(python_version), ver_str(gdal_version),
               ver_str(this_python_version_will_be_deprecated_in_gdal_version),
               ver_str(next_version_of_gdal_will_use_python_version),
               ver_str(this_python_version_will_be_deprecated_in_gdal_version))

    from warnings import warn, simplefilter
    simplefilter('always', DeprecationWarning)
    warn(msg, DeprecationWarning)
