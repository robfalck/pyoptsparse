.. _snopt:

SNOPT
=====
SNOPT is a sparse nonlinear optimizer that is particularly useful for
solving large-scale constrained problems with smooth objective
functions and constraints. The algorithm consists of a sequential
quadratic programming (SQP) algorithm that uses a smooth augmented
Lagrangian merit function, while making explicit provision for
infeasibility in the original problem and in the quadratic programming
subproblems. The Hessian of the Lagrangian is approximated using the
BFGS quasi-Newton update.

Installation
------------

Building from source
********************
SNOPT is available for purchase `here
<http://www.sbsi-sol-optimize.com/asp/sol_snopt.htm>`_. Upon purchase, you should receive a zip file. Within the zip file, there is a folder called ``src``. To use SNOPT with pyoptsparse, paste all files from ``src`` except snopth.f into ``pyoptsparse/pySNOPT/source``.

From v2.0 onwards, only SNOPT v7.7.x is officially supported.
To use pyOptSparse with previous versions of SNOPT, please checkout release v1.2.
We currently test v7.7.7 and v7.7.1.

Installation by conda
*********************

.. _snopt_by_conda:

When installing via conda, all pyoptsparse binaries are pre-compiled and installed as part of the package.
However, the `snopt` binding module cannot be included as part of the package due to license restrictions.

If you are installing via conda and would like to use SNOPT, you will need to build the `snopt` binding module on your own, and inform `pyoptsparse` that it should use that library.

**Automated build tool (recommended)**

pyoptsparse includes a build tool that automates the process of building the SNOPT module. You can either build from source or link against a precompiled library.

**Option 1: Build from source**

Simply provide the path to your SNOPT source directory:

.. code-block:: bash

    python -m pyoptsparse.build_snopt_module /path/to/snopt/src

Or using the console script:

.. code-block:: bash

    pyoptsparse-build-snopt /path/to/snopt/src

This will:

1. Automatically download required build files from GitHub (if not present in your installation)
2. Copy your SNOPT source files to a temporary build directory
3. Build the SNOPT Python extension module using meson (which uses f2py internally)
4. Install it directly into the pyoptsparse package directory (``site-packages/pyoptsparse/pySNOPT/``)
5. **No environment variable configuration needed!** SNOPT will be automatically detected.

.. note::
   The build tool requires ``meson`` and ``ninja`` to be installed. If not already present, install them with:

   .. code-block:: bash

       pip install meson ninja

.. note::
   For conda installations, the required build files (f2py interface, helper scripts) are automatically
   downloaded from the pyoptsparse GitHub repository. An internet connection is required for the first build.

**Option 2: Link against precompiled library**

If you have a precompiled SNOPT library, you can link against it instead:

- Linux: ``libsnopt7.so``
- macOS: ``libsnopt7.dylib``
- Windows: ``snopt7.dll``

.. code-block:: bash

    # Linux/macOS
    python -m pyoptsparse.build_snopt_module --snopt-lib /path/to/libsnopt7.so

    # Windows
    python -m pyoptsparse.build_snopt_module --snopt-lib C:\path\to\snopt7.dll

This creates a lightweight Python wrapper that links against your precompiled library, which is faster than compiling from source. This approach is particularly useful when using SNOPT libraries from conda or other package managers.

.. note::
   When using ``--snopt-lib``, you only need ``meson`` and ``ninja`` installed. No Fortran compiler is required since the library is already compiled.

To test your installation:

.. code-block:: bash

    python -c "from pyoptsparse import SNOPT; print('SNOPT loaded successfully!')"

You can specify a custom output directory with ``--output`` (this will require setting ``PYOPTSPARSE_IMPORT_SNOPT_FROM``):

.. code-block:: bash

    python -m pyoptsparse.build_snopt_module /path/to/snopt/src --output ~/my-snopt
    export PYOPTSPARSE_IMPORT_SNOPT_FROM=~/my-snopt/

For more options, run:

.. code-block:: bash

    python -m pyoptsparse.build_snopt_module --help

**Manual build**

Suppose you have built the binding file, producing ``snopt.cpython-310.so``, living in the folder ``~/snopt-bind``.

To use this module, set the environment variable, ``PYOPTSPARSE_IMPORT_SNOPT_FROM``, e.g.:

.. code-block:: bash

    PYOPTSPARSE_IMPORT_SNOPT_FROM=~/snopt-bind/

This will attempt to load the ``snopt`` binding module from ``~/snopt-bind``. If the module cannot be loaded from this path, a warning will be raised at import time, and an error will be raised if attempting to run the SNOPT optimizer.

Options
-------
Please refer to the SNOPT user manual for a complete listing of options and their default values.
The following are a list of

- options which have values changed from the defaults within SNOPT
- options unique to pyOptSparse, implemented in the Python wrapper and not found in SNOPT

.. optionstable:: pyoptsparse.pySNOPT.pySNOPT.SNOPT
   :filename: SNOPT_options.yaml

Informs
-------
.. optionstable:: pyoptsparse.pySNOPT.pySNOPT.SNOPT
   :type: informs

API
---
.. currentmodule:: pyoptsparse.pySNOPT.pySNOPT

.. autoclass:: SNOPT
   :members: __call__
