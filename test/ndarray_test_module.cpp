#include <boost/python.hpp>
#include <boost/numpy.hpp>

namespace bp = boost::python;
namespace np = boost::numpy;

namespace test {

np::ndarray zeros(bp::tuple shape, np::dtype dt) { return np::zeros(shape, dt); }
np::ndarray array1(bp::object obj) { return np::array(obj); }
np::ndarray array2(bp::object obj, np::dtype dt) { return np::array(obj, dt); }
np::ndarray empty(bp::tuple shape, np::dtype dt) { return np::empty(shape, dt);}
np::ndarray transpose(np::ndarray arr) { return arr.transpose(); }
np::ndarray squeeze(np::ndarray arr) { return arr.squeeze(); }
np::ndarray reshape(np::ndarray arr, bp::tuple shape) { return arr.reshape(shape);}

}// namespace test

BOOST_PYTHON_MODULE(ndarray_test_module)
{
    np::initialize();
    bp::def("zeros",           &test::zeros);
    bp::def("zeros_as_matrix", &test::zeros, np::as_matrix<>());
    bp::def("array",           &test::array1);
    bp::def("array",           &test::array2);
    bp::def("empty",           &test::empty);
    bp::def("transpose",       &test::transpose);
    bp::def("squeeze",         &test::squeeze);
    bp::def("reshape",         &test::reshape);
}
