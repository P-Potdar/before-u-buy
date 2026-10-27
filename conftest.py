import os
import tempfile

_test_database = tempfile.TemporaryDirectory(prefix="beforeyoubuy-pytest-")
os.environ["BEFOREYOUBUY_DB_PATH"] = os.path.join(_test_database.name, "test.sqlite3")


def pytest_sessionfinish(session, exitstatus):
    _test_database.cleanup()
