#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "hik_camera_core.h"

namespace py = pybind11;

PYBIND11_MODULE(hik_camera_cpp, m) {
    m.doc() = "Hikrobot MVS camera Python bindings";

    py::class_<FrameData>(m, "FrameData")
        .def(py::init<>())
        .def_readonly("width", &FrameData::width)
        .def_readonly("height", &FrameData::height)
        .def_readonly("channels", &FrameData::channels)
        .def_readonly("timestamp", &FrameData::timestamp)
        .def_readonly("data", &FrameData::data);

    py::class_<HikCameraCore>(m, "HikCameraCore")
        .def(py::init<const std::string&>(), py::arg("config_path"))
        .def("open", &HikCameraCore::open)
        .def("close", &HikCameraCore::close)
        .def("start", &HikCameraCore::start)
        .def("stop", &HikCameraCore::stop)
        .def("is_opened", &HikCameraCore::is_opened)
        .def("is_grabbing", &HikCameraCore::is_grabbing)
        .def("get_frame", &HikCameraCore::get_frame, py::arg("timeout_ms") = 1000)
        .def("width", &HikCameraCore::width)
        .def("height", &HikCameraCore::height)
        .def("serial_number", &HikCameraCore::serial_number)
        .def("config_path", &HikCameraCore::config_path);
}