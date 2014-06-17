import pytest
import os,sys

wrapped_stdin = sys.stdin
sys.stdin = sys.__stdin__
from IPython.kernel import KernelManager
sys.stdin = wrapped_stdin

from IPython.nbformat.current import reads

# combined from
# http://pytest.org/latest/example/nonpython.html#non-python-tests
# and
# https://gist.github.com/2621679 by minrk

tests = ["parsing", "hypergraphs", "decipher", "Fibonacci", "BuildingHypergraph", "sequence_crf", "BeamSearch", "DFA"]

def pytest_collect_file(path, parent):
    print path
    if path.ext == ".ipynb" and any([t + "." in str(path) for t in tests]):
        return IPyNbFile(path, parent)

class IPyNbFile(pytest.File):
    def collect(self):
        with self.fspath.open() as f:
            self.nb = reads(f.read(), 'json')

            cell_num = 0

            for ws in self.nb.worksheets:
                for cell in ws.cells:
                    if cell.cell_type == "code":
                        yield IPyNbCell(self.name, self, cell_num, cell)
                        cell_num += 1

    def setup(self):
        self.km = KernelManager()
        self.km.start_kernel(stderr=open(os.devnull, 'w'))
        self.kc = self.km.client()
        self.kc.start_channels()
        self.shell = self.kc.shell_channel

    def teardown(self):
        self.km.shutdown_kernel()
        del self.shell
        del self.km

class IPyNbCell(pytest.Item):
    def __init__(self, name, parent, cell_num, cell):
        super(IPyNbCell, self).__init__(name, parent)

        self.cell_num = cell_num
        self.cell = cell

    def runtest(self):
        print "running"
        shell = self.parent.shell
        shell.execute(self.cell.input, allow_stdin=False)
        reply = shell.get_msg(timeout=20)['content']
        if reply['status'] == 'error':
            raise IPyNbException(self.cell_num, self.cell.input, '\n'.join(reply['traceback']))

    def repr_failure(self, excinfo):
        """ called when self.runtest() raises an exception. """
        if isinstance(excinfo.value, IPyNbException):
            return "\n".join([
                "notebook worksheet execution failed",
                " cell %s\n\n"
                "   input: %s\n\n"
                "   raised: %s\n" % excinfo.value.args[0:3],
            ])


    def reportinfo(self):
        return self.fspath, 0, "cell %d" % self.cell_num

class IPyNbException(Exception):
    """ custom exception for error reporting. """
