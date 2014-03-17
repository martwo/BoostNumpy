#include <boost/python.hpp>
#include <boost/python/tuple.hpp>
#include <boost/python/slice.hpp>
#include <boost/numpy.hpp>

namespace bp = boost::python;
namespace np = boost::numpy;

namespace test {

bp::object single(np::ndarray arr, int i) { return arr[i]; }
bp::object slice(np::ndarray arr, bp::slice sl) { return arr[sl]; }
bp::object index_array(np::ndarray arr, np::ndarray idxarr) { return arr[idxarr]; }
bp::object index_array_2d(np::ndarray arr, np::ndarray idxarr1, np::ndarray idxarr2) { return arr[bp::make_tuple(idxarr1, idxarr2)]; }
bp::object index_array_slice(np::ndarray arr, np::ndarray idxarr, bp::slice sl) { return arr[bp::make_tuple(idxarr, sl)]; }

}// namespace test

BOOST_PYTHON_MODULE(indexing_test_module)
{
    np::initialize();
    bp::def("single",            &test::single);
    bp::def("slice",             &test::slice);
    bp::def("index_array",       &test::index_array);
    bp::def("index_array_2d",    &test::index_array_2d);
    bp::def("index_array_slice", &test::index_array_slice);
}
