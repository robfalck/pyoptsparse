"""Test the pickle-free history file storage in pyOpt_sqlite"""

# Standard Python modules
from collections import OrderedDict
import math
import os
import pickle
import shutil
import sqlite3
import tempfile
import unittest

# External modules
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.sparse import csr_array, csr_matrix

# First party modules
from pyoptsparse import OPT, History, Optimization
from pyoptsparse.postprocessing.OptView_baseclass import OVBaseClass
from pyoptsparse.pyOpt_sqlite import MAGIC, SqliteDict, decode, encode

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
# written by pyOptSparse 2.17 with sqlitedict 2.1.0, before sqlitedict was removed
LEGACY_HIST = os.path.join(DATA_DIR, "legacy_sqlitedict.hst")
# the old OpenMDAO recorder layout ("iterations", "derivs" and "metadata" tables) read by OptView
LEGACY_OPENMDAO = os.path.join(DATA_DIR, "legacy_openmdao_recorder.db")

# the problem used to generate LEGACY_HIST
N = 4


def objfunc(xdict):
    x = xdict["xvars"]
    funcs = {"obj": sum(100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2 for i in range(N - 1))}
    funcs["con"] = np.array([x[0] ** 2 + x[1] ** 2])
    return funcs, False


def sens(xdict, funcs):
    x = xdict["xvars"]
    g = np.zeros(N)
    for i in range(N - 1):
        g[i] += 2 * (200 * x[i] ** 3 - 200 * x[i] * x[i + 1] + x[i] - 1)
        g[i + 1] += 200 * (x[i + 1] - x[i] ** 2)
    dcon = np.zeros((1, N))
    dcon[0, 0] = 2 * x[0]
    dcon[0, 1] = 2 * x[1]
    return {"obj": {"xvars": g}, "con": {"xvars": dcon}}, False


def make_optProb():
    optProb = Optimization("Rosenbrock legacy fixture", objfunc)
    optProb.addVarGroup("xvars", N, lower=-5.0, upper=5.0, value=np.array([-1.2, 1.0, -0.5, 0.8]))
    optProb.addObj("obj")
    optProb.addConGroup("con", 1, upper=4.0)
    optProb.addConGroup("lin", 1, lower=None, upper=10.0, linear=True, jac={"xvars": np.ones((1, N))}, wrt=["xvars"])
    return optProb


def make_stored_optProb():
    """An optProb as the optimizer stores it, with the user callables removed."""
    optProb = make_optProb()
    optProb.finalize()
    optProb.objFun = None
    optProb.sens = None
    return optProb


def roundtrip(obj):
    return decode(encode(obj))


class TestCodec(unittest.TestCase):
    def assertRoundTrip(self, obj):
        out = roundtrip(obj)
        self.assertIs(type(out), type(obj))
        self.assertEqual(out, obj)

    def test_scalars(self):
        for obj in [None, True, False, 0, -1, 2**63 - 1, -(2**63), 1.5, -0.0, math.inf, -math.inf, 1 + 2j, "", "héllo"]:
            with self.subTest(obj=obj):
                self.assertRoundTrip(obj)
        self.assertRoundTrip(b"\x00\x80bytes")
        self.assertTrue(math.isnan(roundtrip(math.nan)))
        self.assertEqual(math.copysign(1.0, roundtrip(-0.0)), -1.0)

    def test_containers(self):
        for obj in [[], (), {}, [1, (2, 3.0), {"a": [None, "b"]}], {"k": {"nested": (1,)}, 3: "int key"}]:
            with self.subTest(obj=obj):
                self.assertRoundTrip(obj)

    def test_ordered_dict_order(self):
        obj = OrderedDict([("z", 1), ("a", 2), ("m", 3)])
        out = roundtrip(obj)
        self.assertEqual(list(out.items()), list(obj.items()))

    def test_arrays(self):
        arrays = [
            np.arange(12, dtype=float).reshape(3, 4),
            np.array([1 + 2j, 3 - 4j]),
            np.array(3.5),
            np.zeros((0, 3)),
            np.arange(10, dtype=np.int32),
            np.array([True, False]),
            np.array(["ab", "c"]),
            np.arange(20.0).reshape(4, 5)[:, ::2],  # non-contiguous
            np.asfortranarray(np.arange(6.0).reshape(2, 3)),
            np.arange(4, dtype=">f8"),  # non-native byte order
        ]
        for arr in arrays:
            with self.subTest(arr=arr):
                out = roundtrip(arr)
                self.assertIsInstance(out, np.ndarray)
                self.assertEqual(out.dtype, arr.dtype)
                self.assertEqual(out.shape, arr.shape)
                assert_array_equal(out, arr)
                self.assertTrue(out.flags.writeable)

    def test_object_arrays(self):
        # constraint bounds are stored like this, e.g. array([None], dtype=object)
        for arr in [np.array([None, 1.0, -2], dtype=object), np.array([[None, "a"], [3, None]], dtype=object)]:
            with self.subTest(arr=arr):
                out = roundtrip(arr)
                self.assertEqual(out.dtype, object)
                self.assertEqual(out.shape, arr.shape)
                self.assertEqual(out.tolist(), arr.tolist())

    def test_numpy_scalars(self):
        for obj, expected in [
            (np.float64(1.5), 1.5),
            (np.int32(-3), -3),
            (np.bool_(True), True),
            (np.complex128(1j), 1j),
        ]:
            with self.subTest(obj=obj):
                out = roundtrip(obj)
                self.assertIs(type(out), type(expected))
                self.assertEqual(out, expected)

    def test_csr(self):
        dense = np.array([[1.0, 0.0, 2.0], [0.0, 0.0, 3.0]])
        for cls in (csr_matrix, csr_array):
            with self.subTest(cls=cls):
                out = roundtrip(cls(dense))
                self.assertIs(type(out), cls)
                assert_array_equal(out.toarray(), dense)

    def test_unsupported_types(self):
        for obj in [lambda x: x, {1, 2}, object(), np.array([len], dtype=object), np.zeros(2, dtype="f8,i4"), 2**64]:
            with self.subTest(obj=obj), self.assertRaises(TypeError):
                encode(obj)

    def test_corrupt_data(self):
        blob = encode({"x": np.arange(5.0)})
        for bad in [blob[:-3], blob[: len(MAGIC)], blob + b"extra", MAGIC + b"?"]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                decode(bad)

    def test_new_format_is_not_pickle(self):
        blob = encode({"x": np.arange(3.0), "optProb": make_stored_optProb()})
        self.assertTrue(blob.startswith(MAGIC))
        with self.assertRaises(pickle.UnpicklingError):
            pickle.loads(blob)


class TestObjects(unittest.TestCase):
    def setUp(self):
        self.optProb = make_stored_optProb()

    def test_components(self):
        var = self.optProb.variables["xvars"][0]
        self.assertEqual(roundtrip(var), var)

        obj = roundtrip(self.optProb.objectives["obj"])
        self.assertEqual(obj.__dict__, self.optProb.objectives["obj"].__dict__)

        con = self.optProb.constraints["lin"]
        out = roundtrip(con)
        self.assertEqual(out.wrt, con.wrt)
        self.assertEqual(out.lower.tolist(), [None])
        assert_array_equal(out.upper, con.upper)
        self.assertIsInstance(out.linearJacobian, csr_matrix)
        assert_array_equal(out.linearJacobian.toarray(), con.linearJacobian.toarray())

    def test_optimization(self):
        out = roundtrip(self.optProb)
        self.assertIsInstance(out, Optimization)
        self.assertNotIn("comm", out.__dict__)
        self.assertEqual(list(out.variables), list(self.optProb.variables))
        self.assertEqual(list(out.constraints), ["con", "lin"])
        assert_allclose(out.getDVs()["xvars"], self.optProb.getDVs()["xvars"])
        assert_allclose(out.invXScale, self.optProb.invXScale)


class TestLegacyPickles(unittest.TestCase):
    def test_allowed(self):
        obj = {"x": np.arange(3.0), "info": OrderedDict(a=1), "c": 1j, "s": np.float64(2.0)}
        out = decode(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
        assert_array_equal(out["x"], obj["x"])
        self.assertEqual(out["info"], obj["info"])
        self.assertEqual(out["c"], 1j)
        self.assertEqual(out["s"], 2.0)

    def test_refused(self):
        executed = []

        class Exploit:
            def __reduce__(self):
                return (executed.append, ("pwned",))

        class EvalExploit:
            def __reduce__(self):
                return (eval, ("1 + 1",))

        for payload in [Exploit(), EvalExploit(), {"nested": [Exploit()]}]:
            with self.subTest(payload=payload), self.assertRaises(pickle.UnpicklingError):
                decode(pickle.dumps(payload))
        self.assertEqual(executed, [])


class TestSqliteDict(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.fileName = os.path.join(self.tmp.name, "test.db")

    def tearDown(self):
        self.tmp.cleanup()

    def test_mapping(self):
        with SqliteDict(self.fileName) as db:
            db["b"] = 1
            db["a"] = np.arange(3.0)
            db["c"] = {"x": 2}
            db["b"] = "updated"  # updating keeps the original position
            self.assertEqual(list(db), ["b", "a", "c"])
            self.assertEqual(len(db), 3)
            self.assertIn("a", db)
            self.assertNotIn("missing", db)
            self.assertEqual(db["b"], "updated")
            with self.assertRaises(KeyError):
                db["missing"]
            del db["c"]
            with self.assertRaises(KeyError):
                del db["c"]
            self.assertEqual([k for k, _ in db.items()], ["b", "a"])

        with SqliteDict(self.fileName) as db:
            self.assertEqual(list(db), ["b", "a"])
            assert_array_equal(db["a"], np.arange(3.0))

    def test_uncommitted_changes_are_dropped(self):
        db = SqliteDict(self.fileName)
        db["kept"] = 1
        db.commit()
        db["dropped"] = 2
        db._conn.rollback()
        db.close()
        with SqliteDict(self.fileName) as db:
            self.assertEqual(list(db), ["kept"])

    def test_tables(self):
        with SqliteDict(self.fileName) as db:
            db["key"] = "default table"
        with SqliteDict(self.fileName, "other") as db:
            self.assertEqual(len(db), 0)
            db["key"] = "other table"
        with SqliteDict(self.fileName) as db:
            self.assertEqual(db["key"], "default table")
        for bad in ['x"; DROP TABLE unnamed; --', "", "1abc"]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                SqliteDict(self.fileName, bad)

    def test_sqlitedict_layout(self):
        with SqliteDict(self.fileName) as db:
            db["k"] = 1
        conn = sqlite3.connect(self.fileName)
        try:
            self.assertEqual(
                conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall(), [("unnamed",)]
            )
            self.assertEqual(conn.execute("SELECT key FROM unnamed").fetchall(), [("k",)])
        finally:
            conn.close()


class TestLegacyFiles(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)

    def tearDown(self):
        self.tmp.cleanup()

    def test_read_history(self):
        hist = History(LEGACY_HIST)
        self.assertEqual(hist.getDVNames(), ["xvars"])
        self.assertEqual(sorted(hist.getConNames()), ["con"])
        self.assertIsInstance(hist.getOptProb(), Optimization)
        self.assertIsInstance(hist.getOptProb().constraints["lin"].linearJacobian, csr_matrix)
        vals = hist.getValues(names=["xvars", "obj"], major=False)
        assert_allclose(vals["xvars"][-1], np.ones(N), atol=1e-4)
        self.assertLess(vals["obj"][-1, 0], 1e-8)

    def test_hotstart_from_legacy(self):
        hotStart = os.path.join(self.tmp.name, "legacy.hst")
        shutil.copy(LEGACY_HIST, hotStart)
        newHist = os.path.join(self.tmp.name, "new.hst")
        opt = OPT("SLSQP", options={"IPRINT": -1, "ACC": 1e-8})
        sol = opt(make_optProb(), sens=sens, storeHistory=newHist, hotStart=hotStart)
        assert_allclose(sol.xStar["xvars"], np.ones(N), atol=1e-4)
        # every evaluation was replayed from the legacy file
        self.assertEqual(sol.userObjCalls, 0)

        # metadata copied from the legacy file is re-encoded in the new format
        conn = sqlite3.connect(newHist)
        try:
            blobs = dict(conn.execute("SELECT key, value FROM unnamed").fetchall())
        finally:
            conn.close()
        for key in ["optProb", "varInfo", "metadata", "0", "last"]:
            self.assertTrue(blobs[key].startswith(MAGIC), key)

        hist = History(newHist)
        xvars = hist.getValues(names=["xvars"], major=False)["xvars"]
        assert_allclose(xvars[-1], sol.xStar["xvars"])

    def test_optview_pyoptsparse(self):
        ov = _OptViewLoader([LEGACY_HIST])
        self.assertIn("obj", ov.func_data_all)
        self.assertIn("xvars", ov.var_data_all)
        self.assertIn("xvars", ov.bounds)

    def test_optview_openmdao(self):
        ov = _OptViewLoader([LEGACY_OPENMDAO])
        # only iteration 2 has derivative info, so it is the only major iteration
        self.assertEqual(sorted(ov.func_data_all), ["con (c)", "obj (o)"])
        self.assertEqual(sorted(ov.var_data_all), ["p", "x (dv)"])
        assert_array_equal(ov.func_data_all["obj (o)"][0], [5.0])
        assert_array_equal(ov.func_data_all["con (c)"][0], [2.0, 4.0])
        assert_array_equal(ov.var_data_all["x (dv)"][0], [2.0])


class _OptViewLoader(OVBaseClass):
    """The parts of OptView needed to call OptimizationHistory() without a GUI."""

    def __init__(self, histList):
        self.histList = histList
        self.bounds = {}
        self.scaling = {}
        self.warnings = []
        self.OptimizationHistory()

    def warning_display(self, string=""):
        self.warnings.append(string)


if __name__ == "__main__":
    unittest.main()
