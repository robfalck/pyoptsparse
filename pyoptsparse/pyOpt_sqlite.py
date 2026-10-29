"""
A minimal, pickle-free replacement for ``sqlitedict.SqliteDict``.

Values are stored in an SQLite table using the same layout as ``sqlitedict``
(``key TEXT PRIMARY KEY, value BLOB`` in a table named ``unnamed`` by default),
so existing history files can be opened without any migration.

New values are serialized with a small typed binary codec that never calls
``pickle``. Values written by ``sqlitedict`` (which pickles everything) are
still readable, but only through a restricted unpickler that refuses to load
any class outside a fixed allowlist. This avoids the arbitrary-code-execution
risk of CVE-2024-35515 when opening an untrusted file.
"""

# Standard Python modules
from collections.abc import Iterator, MutableMapping
import io
import math
import pickle
import re
import sqlite3
import struct
from typing import Any

# External modules
import numpy as np
from scipy.sparse import csr_matrix

__all__ = ["SqliteDict", "encode", "decode"]

# Prefix for blobs written by this module: 4-byte tag + 1-byte format version.
# Pickled blobs always start with b"\x80" (protocol >= 2), so they cannot collide.
MAGIC = b"PYOS\x01"

# Type tags
_NONE = b"N"
_BOOL = b"B"
_INT = b"i"
_FLOAT = b"f"
_COMPLEX = b"c"
_STR = b"s"
_LIST = b"l"
_TUPLE = b"t"
_DICT = b"d"
_NDARRAY = b"a"
_OBJARRAY = b"O"
_CSR = b"r"
_OBJECT = b"o"

_U32 = struct.Struct(">I")
_I64 = struct.Struct(">q")
_F64 = struct.Struct(">d")
_C128 = struct.Struct(">dd")

_TABLENAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# Classes that may be stored with the OBJECT tag. Populated lazily, since these
# modules import this one.
_OBJECT_CLASSES: dict[str, type] = {}


def _object_classes() -> dict[str, type]:
    if not _OBJECT_CLASSES:
        from .pyOpt_constraint import Constraint
        from .pyOpt_objective import Objective
        from .pyOpt_optimization import Optimization
        from .pyOpt_variable import Variable

        _OBJECT_CLASSES.update(
            {"Optimization": Optimization, "Variable": Variable, "Constraint": Constraint, "Objective": Objective}
        )
    return _OBJECT_CLASSES


# ----------------------------------------------------------------------------
# Encoding
# ----------------------------------------------------------------------------


def encode(obj: Any) -> bytes:
    """
    Serialize ``obj`` without pickle. Supported types are ``None``, ``bool``, ``int``,
    ``float``, ``complex``, ``str``, ``list``, ``tuple``, ``dict``, numpy arrays and
    scalars, ``scipy.sparse.csr_matrix``, and the pyOptSparse ``Optimization``,
    ``Variable``, ``Constraint`` and ``Objective`` classes. Anything else raises ``TypeError``.
    """
    buf = bytearray(MAGIC)
    _encode(obj, buf)
    return bytes(buf)


def _write_bytes(data: bytes, buf: bytearray) -> None:
    buf += _U32.pack(len(data))
    buf += data


def _encode(obj: Any, buf: bytearray) -> None:
    if obj is None:
        buf += _NONE
    elif isinstance(obj, bool):
        buf += _BOOL + (b"\x01" if obj else b"\x00")
    elif isinstance(obj, np.generic):
        # numpy scalars are stored as the equivalent Python value. Extended-precision
        # types (longdouble on most platforms) have no Python equivalent and are rejected.
        item = obj.item()
        if isinstance(item, np.generic):
            raise TypeError(f"Cannot store numpy scalar of type {type(obj).__name__}")
        _encode(item, buf)
    elif isinstance(obj, int):
        try:
            buf += _INT + _I64.pack(obj)
        except struct.error:
            raise TypeError(f"Integer {obj} does not fit in 64 bits") from None
    elif isinstance(obj, float):
        buf += _FLOAT + _F64.pack(obj)
    elif isinstance(obj, complex):
        buf += _COMPLEX + _C128.pack(obj.real, obj.imag)
    elif isinstance(obj, str):
        buf += _STR
        _write_bytes(obj.encode("utf-8"), buf)
    elif isinstance(obj, list | tuple):
        buf += _LIST if isinstance(obj, list) else _TUPLE
        buf += _U32.pack(len(obj))
        for item in obj:
            _encode(item, buf)
    elif isinstance(obj, dict):
        buf += _DICT + _U32.pack(len(obj))
        for key, value in obj.items():
            _encode(key, buf)
            _encode(value, buf)
    elif isinstance(obj, np.ndarray) and obj.dtype == object:
        # e.g. constraint bounds such as array([None, 1.0], dtype=object);
        # each element is encoded with this codec, never as raw memory
        buf += _OBJARRAY + _U32.pack(obj.ndim)
        for n in obj.shape:
            buf += _U32.pack(n)
        for item in obj.ravel(order="C"):
            _encode(item, buf)
    elif isinstance(obj, np.ndarray):
        buf += _NDARRAY
        _encode_array(obj, buf)
    elif isinstance(obj, csr_matrix):
        buf += _CSR
        buf += _U32.pack(obj.shape[0]) + _U32.pack(obj.shape[1])
        for arr in (obj.data, obj.indices, obj.indptr):
            _encode_array(arr, buf)
    elif type(obj).__name__ in _object_classes() and _object_classes()[type(obj).__name__] is type(obj):
        state = obj.__getstate__()
        if not isinstance(state, dict):
            raise TypeError(f"Cannot store {type(obj).__name__}: __getstate__ did not return a dict")
        buf += _OBJECT
        _write_bytes(type(obj).__name__.encode("utf-8"), buf)
        _encode(state, buf)
    else:
        raise TypeError(f"Cannot store object of type {type(obj).__module__}.{type(obj).__qualname__}")


def _check_dtype(dtype: np.dtype) -> None:
    if dtype.hasobject or dtype.kind == "V":
        raise TypeError(f"Cannot store numpy array with dtype {dtype}")
    if (dtype.kind == "f" and dtype.itemsize > 8) or (dtype.kind == "c" and dtype.itemsize > 16):
        raise TypeError(f"Cannot store extended-precision numpy array with dtype {dtype}")


def _encode_array(arr: np.ndarray, buf: bytearray) -> None:
    _check_dtype(arr.dtype)
    # dtype.str includes the byte order (e.g. "<f8"), so files are portable
    _write_bytes(arr.dtype.str.encode("ascii"), buf)
    buf += _U32.pack(arr.ndim)
    for n in arr.shape:
        buf += _U32.pack(n)
    _write_bytes(arr.tobytes(order="C"), buf)


# ----------------------------------------------------------------------------
# Decoding
# ----------------------------------------------------------------------------


class _Reader:
    def __init__(self, data: bytes):
        self.data = memoryview(data)
        self.pos = 0

    def read(self, n: int) -> memoryview:
        end = self.pos + n
        if end > len(self.data):
            raise ValueError("unexpected end of data")
        out = self.data[self.pos : end]
        self.pos = end
        return out

    def u32(self) -> int:
        return _U32.unpack(self.read(4))[0]

    def blob(self) -> bytes:
        return bytes(self.read(self.u32()))


def decode(blob: bytes, key: str | None = None) -> Any:
    """
    Deserialize a value written by :func:`encode`, or by ``sqlitedict``. Corrupt data
    raises ``ValueError`` (naming ``key`` if given), and a legacy pickle that references a
    class outside the allowlist raises ``pickle.UnpicklingError``.
    """
    blob = bytes(blob)
    if not blob.startswith(MAGIC):
        return _RestrictedUnpickler(io.BytesIO(blob)).load()

    reader = _Reader(blob)
    reader.pos = len(MAGIC)
    try:
        obj = _decode(reader)
        if reader.pos != len(blob):
            raise ValueError("trailing data after value")
    except (ValueError, TypeError, KeyError, struct.error, UnicodeDecodeError, RecursionError) as e:
        where = f" for key '{key}'" if key is not None else ""
        raise ValueError(f"Corrupt history data{where}: {e}") from e
    return obj


def _decode(r: _Reader) -> Any:
    tag = bytes(r.read(1))
    if tag == _NONE:
        return None
    if tag == _BOOL:
        return bytes(r.read(1)) != b"\x00"
    if tag == _INT:
        return _I64.unpack(r.read(8))[0]
    if tag == _FLOAT:
        return _F64.unpack(r.read(8))[0]
    if tag == _COMPLEX:
        return complex(*_C128.unpack(r.read(16)))
    if tag == _STR:
        return r.blob().decode("utf-8")
    if tag in (_LIST, _TUPLE):
        items = [_decode(r) for _ in range(r.u32())]
        return items if tag == _LIST else tuple(items)
    if tag == _DICT:
        out = {}
        for _ in range(r.u32()):
            k = _decode(r)
            out[k] = _decode(r)
        return out
    if tag == _NDARRAY:
        return _decode_array(r)
    if tag == _OBJARRAY:
        shape = tuple(r.u32() for _ in range(r.u32()))
        # every element takes at least one byte, so this bounds the allocation
        if math.prod(shape) > len(r.data) - r.pos:
            raise ValueError("array shape exceeds the remaining data")
        arr = np.empty(shape, dtype=object)
        flat = arr.reshape(-1)  # a view, since arr is freshly allocated and contiguous
        for i in range(flat.size):
            flat[i] = _decode(r)
        return arr
    if tag == _CSR:
        shape = (r.u32(), r.u32())
        data, indices, indptr = (_decode_array(r) for _ in range(3))
        return csr_matrix((data, indices, indptr), shape=shape)
    if tag == _OBJECT:
        name = r.blob().decode("utf-8")
        classes = _object_classes()
        if name not in classes:
            raise ValueError(f"unknown object type '{name}'")
        state = _decode(r)
        if not isinstance(state, dict) or not all(isinstance(k, str) for k in state):
            raise ValueError(f"invalid state for '{name}'")
        cls = classes[name]
        obj = cls.__new__(cls)
        if hasattr(cls, "__setstate__"):
            obj.__setstate__(state)
        else:
            obj.__dict__.update(state)
        return obj
    raise ValueError(f"unknown type tag {tag!r}")


def _decode_array(r: _Reader) -> np.ndarray:
    dtype = np.dtype(r.blob().decode("ascii"))
    _check_dtype(dtype)
    shape = tuple(r.u32() for _ in range(r.u32()))
    data = r.blob()
    if len(data) != math.prod(shape) * dtype.itemsize:
        raise ValueError("array data does not match its shape")
    return np.frombuffer(data, dtype=dtype).reshape(shape).copy()


class _RestrictedUnpickler(pickle.Unpickler):
    """
    Unpickler for values written by ``sqlitedict``, limited to the classes
    pyOptSparse has actually stored in history files.
    """

    _ALLOWED = {
        "builtins": {"complex", "set", "frozenset", "slice"},
        "collections": {"OrderedDict"},
        "numpy": {"ndarray", "dtype"},
        "numpy.core.multiarray": {"_reconstruct", "scalar"},
        "numpy._core.multiarray": {"_reconstruct", "scalar"},
        "numpy.core.numeric": {"_frombuffer"},
        "numpy._core.numeric": {"_frombuffer"},
        "scipy.sparse._csr": {"csr_matrix"},
        "scipy.sparse.csr": {"csr_matrix"},
        "pyoptsparse.pyOpt_optimization": {"Optimization"},
        "pyoptsparse.pyOpt_variable": {"Variable"},
        "pyoptsparse.pyOpt_constraint": {"Constraint"},
        "pyoptsparse.pyOpt_objective": {"Objective"},
    }

    def find_class(self, module: str, name: str) -> Any:
        if name in self._ALLOWED.get(module, ()):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"Refusing to load {module}.{name} from legacy history file")


# ----------------------------------------------------------------------------
# Mapping
# ----------------------------------------------------------------------------


class SqliteDict(MutableMapping):
    def __init__(self, filename: str, tablename: str = "unnamed"):
        """
        A dictionary stored in an SQLite database, compatible with files written by ``sqlitedict``.

        Changes are not written to disk until :meth:`commit` (or :meth:`close`) is called.

        Parameters
        ----------
        filename : str
            Path to the database file. It is created if it does not exist.
        tablename : str
            Name of the table holding the dictionary. Files written by pyOptSparse use
            the default, ``"unnamed"``.
        """
        if not isinstance(tablename, str) or not _TABLENAME_RE.fullmatch(tablename):
            raise ValueError(f"Invalid table name '{tablename}'")
        self.filename = filename
        self.tablename = tablename
        self._db = sqlite3.connect(filename)
        self._db.execute(f'CREATE TABLE IF NOT EXISTS "{tablename}" (key TEXT PRIMARY KEY, value BLOB)')
        self._db.commit()

    def __getitem__(self, key: str) -> Any:
        row = self._db.execute(f'SELECT value FROM "{self.tablename}" WHERE key = ?', (key,)).fetchone()
        if row is None:
            raise KeyError(key)
        return decode(row[0], key)

    def __setitem__(self, key: str, value: Any) -> None:
        # upsert keeps the original rowid, so iteration order is insertion order
        self._db.execute(
            f'INSERT INTO "{self.tablename}" (key, value) VALUES (?, ?) '
            + "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, encode(value)),
        )

    def __delitem__(self, key: str) -> None:
        if self._db.execute(f'DELETE FROM "{self.tablename}" WHERE key = ?', (key,)).rowcount == 0:
            raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        for (key,) in self._db.execute(f'SELECT key FROM "{self.tablename}" ORDER BY rowid').fetchall():
            yield key

    def __len__(self) -> int:
        return self._db.execute(f'SELECT COUNT(*) FROM "{self.tablename}"').fetchone()[0]

    def __contains__(self, key: object) -> bool:
        return self._db.execute(f'SELECT 1 FROM "{self.tablename}" WHERE key = ?', (key,)).fetchone() is not None

    def items(self) -> list[tuple[str, Any]]:
        """Return all ``(key, value)`` pairs in insertion order, using a single query."""
        rows = self._db.execute(f'SELECT key, value FROM "{self.tablename}" ORDER BY rowid').fetchall()
        return [(key, decode(value, key)) for key, value in rows]

    def commit(self) -> None:
        """Write pending changes to disk."""
        self._db.commit()

    sync = commit

    def close(self) -> None:
        """Commit pending changes and close the database."""
        self._db.commit()
        self._db.close()

    def __enter__(self) -> "SqliteDict":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
