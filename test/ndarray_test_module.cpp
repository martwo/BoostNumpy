#include <boost/python.hpp>
#include <boost/numpy.hpp>

namespace bp = boost::python;
namespace np = boost::numpy;

namespace test {

np::ndarray zeros(bp::tuple shape, np::dtype dt) { return np::zeros(shape, dt); }
np::ndarray array1(bp::object obj) { return np::array(obj); }
np::ndarray array2(bp::object obj, np::dtype dt) { return np::array(obj,dt); }

}// namespace test

BOOST_PYTHON_MODULE(ndarray_test_module)
{
    np::initialize();
    bp::def("zeros", &test::zeros);
    bp::def("array", &test::array1);
    bp::def("array", &test::array2);
}
