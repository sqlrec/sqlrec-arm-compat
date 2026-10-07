import graphlearn as gl
import pytest
from graphlearn.python.data.values import Values
import graphlearn.python.data.values as values_module
import graphlearn.python as python_module
import graphlearn.python.data as data_module
from importlib.metadata import version

from sqlrec_arm_compat import UnsupportedAPIError


def test_graphlearn_import_surface_and_explicit_errors():
    assert version("pyfg") == "1.0.5.post2"
    assert version("graphlearn") == "1.3.8.post1"
    assert gl.Graph.__name__ == "Graph"
    assert gl.Nodes.__name__ == "Nodes"
    Values.string_attrs = property(lambda self: None)
    with pytest.raises(UnsupportedAPIError, match="graphlearn.Graph"):
        gl.Graph()
    with pytest.raises(UnsupportedAPIError, match="graphlearn.Decoder"):
        gl.Decoder(weighted=True)
    with pytest.raises(UnsupportedAPIError, match="Values"):
        Values()
    with pytest.raises(UnsupportedAPIError, match="set_tracker_mode"):
        gl.set_tracker_mode(0)
    with pytest.raises(UnsupportedAPIError, match="graphlearn.unknown_api"):
        gl.unknown_api()
    with pytest.raises(UnsupportedAPIError, match="values.unknown_api"):
        values_module.unknown_api()


@pytest.mark.parametrize("name", ["Graph", "Nodes", "Decoder", "set_field_delimiter",
                                  "set_use_string_hash_id", "set_load_graph_thread_num", "set_tracker_mode"])
def test_graphlearn_calls_are_explicitly_unavailable(name):
    with pytest.raises(UnsupportedAPIError, match=name):
        getattr(gl, name)()


@pytest.mark.parametrize("module", [gl, python_module, data_module, values_module])
def test_graphlearn_namespace_errors_preserve_dunder_contract(module):
    with pytest.raises(UnsupportedAPIError, match="unknown_api"):
        module.unknown_api
    with pytest.raises(AttributeError):
        module.__unknown_api__
