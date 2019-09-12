#ifndef GLEW_STATIC
#define GLEW_STATIC
#endif
#include <GL/glew.h>
#include <gl/GL.h>
//#include <GL/glext.h>

#ifdef _WINDOWS
#include <GL/wglew.h>
#elif defined(_LINUX)
#include <GL/xglew.h>
#endif
#include <GLFW/glfw3.h>

#pragma comment(lib, "opengl32")
#pragma comment(lib, "glew")
#pragma comment(lib, "glfw")

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <ctime>

#include <algorithm>
#include <array>
#include <chrono>
#include <fstream>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <iomanip>
#include <list>
#include <memory>
#include <mutex>
#include <queue>
#include <random>
#include <set>
#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <streambuf>
#include <thread>
#include <vector>

#if !defined(__ANDROID__)

//#include <glad/glad.h>

namespace gl {
void init();
GLuint loadShader(const std::string& shaderSource, GLenum shaderType);
GLuint buildProgram(const std::string& vertexShaderSource, const std::string& fragmentShaderSource);
void report();
void setupDebugLogging();
}  // namespace gl

#endif
