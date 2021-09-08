from distutils import dir_util
from pytest import fixture
from pathlib import Path


@fixture
def datadir(tmp_path, request):
    '''
    Fixture responsible for searching a folder with the same name of test
    module and, if available, moving all contents to a temporary directory so
    tests can use them freely.
    '''
    test_file = Path(request.fspath)
    test_dir = test_file.parent
    test_name = test_file.name[5:-3]
    test_dir = test_dir / "fixtures" / test_name

    print(f"test_dir: {test_dir.resolve()}, tmp_path: {tmp_path.resolve()}")
    if test_dir.is_dir():
        dir_util.copy_tree(f"{test_dir.resolve()}", f"{tmp_path.resolve()}")

    return tmp_path