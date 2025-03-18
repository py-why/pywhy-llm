import os
import subprocess
import tempfile

import nbformat
import pytest
from pytest import mark

NOTEBOOKS_PATH = "docs/notebooks/"
notebooks_list = [f.name for f in os.scandir(NOTEBOOKS_PATH) if f.name.endswith(".ipynb")]

advanced_notebooks = {
}

ignore_notebooks = [
]

# Adding the root folder to the python path so that jupyter notebooks
# can import pywhyllm
if "PYTHONPATH" not in os.environ:
    os.environ["PYTHONPATH"] = os.getcwd()
elif os.getcwd() not in os.environ["PYTHONPATH"].split(os.pathsep):
    os.environ["PYTHONPATH"] = os.environ["PYTHONPATH"] + os.pathsep + os.getcwd()


def _notebook_run(filepath):
    """Execute a notebook via nbconvert and collect output.
    :returns (parsed nb object, execution errors)

    Source of this function: http://www.christianmoscardi.com/blog/2016/01/20/jupyter-testing.html
    """
    with tempfile.NamedTemporaryFile(suffix=".ipynb") as fout:
        args = [
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            #          "--ExecutePreprocessor.timeout=600",
            "-y",
            "--no-prompt",
            "--output",
            fout.name,
            filepath,
        ]
        subprocess.check_call(args)

        fout.seek(0)
        nb = nbformat.read(fout, nbformat.current_nbformat)

    errors = [
        output for cell in nb.cells if "outputs" in cell for output in cell["outputs"] if output.output_type == "error"
    ]

    return nb, errors

parameter_list = []
for nb in notebooks_list:
    if nb in ignore_notebooks:
        continue
    else:
        marks = [mark.notebook]
        if nb in advanced_notebooks:
            marks.append(mark.advanced)
        param = pytest.param(nb, marks=marks, id=nb)
    parameter_list.append(param)


@mark.parametrize("notebook_filename", parameter_list)
def test_notebook(notebook_filename):
    nb, errors = _notebook_run(NOTEBOOKS_PATH + notebook_filename)
    assert errors == []
