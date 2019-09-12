# C++ Vulkan examples and demos

<img src="./documentation/images/vulkanlogoscene.png" alt="Vulkan demo scene" height="256px"><img src="./documentation/images/c_0.jpg" alt="C++" height="256px">

A fork of [jherico](https://github.com/jherico) modified Vulkan Examples. The README from the origin repo follows below:

This is a fork of [Sascha Willems](https://github.com/SaschaWillems) excellent [Vulkan examples](https://github.com/SaschaWillems/Vulkan) with some modifications.  

* All of the code except for the VulkanDebug stuff has been ported to use the [Vulkan C++ API](https://github.com/KhronosGroup/Vulkan-Hpp)
* All platform specific code for Windows and Linux has been consolidated to use [GLFW 3.2](http://www.glfw.org/)
* Project files for Visual Studio have been removed in favor of a pure [CMake](https://cmake.org/) based system
* Binary files have been removed in favor of CMake external projects
* Enable validation layers by default when building in debug mode
* Avoid excessive use of vkDeviceWaitIdle and vkQueueWaitIdle
* Avoid excessive use of explicit image layout transitions, instead using implicit transitions via the RenderPass and Subpass definitions

## Known issues

* I've only tested so far on Windows using VS 2013, 2015 & VS 2017.  
* I'm still cleaning up after the migration to `Vulkan.hpp` so the code isn't as clean as it could be.  Lots of unnecessary function parameters and structure assignments remain

# Building

Use the provided CMakeLists.txt for use with [CMake](https://cmake.org) to generate a build configuration for your toolchain.  Using 64 bit builds is strongly recommended. 

# Examples 

This information comes from the [original repository readme](https://github.com/SaschaWillems/Vulkan/blob/master/README.md)

## [Beginner Examples](EXAMPLES_INIT.md)

## [Basic Technique Examples](EXAMPLES_BASIC.md)

## [Offscreen Rendering Examples](EXAMPLES_OFFSCREEN.md)

## [VR Examples](EXAMPLES_VR.md)

## [Compute Examples](EXAMPLES_COMPUTE.md)

## [Broken Examples](EXAMPLES_BROKEN.md) 

# Credits

This information comes from the [original repository readme](https://github.com/SaschaWillems/Vulkan/blob/master/README.md)

Thanks to the authors of these libraries :
- [OpenGL Mathematics (GLM)](https://github.com/g-truc/glm)
- [OpenGL Image (GLI)](https://github.com/g-truc/gli)
- [Open Asset Import Library](https://github.com/assimp/assimp)
- [Tiny obj loader](https://github.com/syoyo/tinyobjloader)

And a huge thanks to the Vulkan Working Group, Vulkan Advisory Panel, the fine people at [LunarG](http://www.lunarg.com), Baldur Karlsson ([RenderDoc](https://github.com/baldurk/renderdoc)) and everyone from the different IHVs that helped me get the examples up and working on their hardware!

## Attributions / Licenses
Please note that (some) models and textures use separate licenses. Please comply to these when redistributing or using them in your own projects :
- Cubemap used in cubemap example by [Emil Persson(aka Humus)](http://www.humus.name/)
- Armored knight model used in deferred example by [Gabriel Piacenti](http://opengameart.org/users/piacenti)
- Voyager model by [NASA](http://nasa3d.arc.nasa.gov/models)
- Astroboy COLLADA model copyright 2008 Sony Computer Entertainment Inc.
- Old deer model used in tessellation example by [Čestmír Dammer](http://opengameart.org/users/cdmir)
- Hidden treasure scene used in pipeline and debug marker examples by [Laurynas Jurgila](http://www.blendswap.com/user/PigArt)
- Textures used in some examples by [Hugues Muller](http://www.yughues-folio.com)
- Updated compute particle system shader by [Lukas Bergdoll](https://github.com/Voultapher)
- Vulkan scene model (and derived models) by [Dominic Agoro-Ombaka](http://www.agorodesign.com/) and [Sascha Willems](http://www.saschawillems.de)
- Vulkan and the Vulkan logo are trademarks of the [Khronos Group Inc.](http://www.khronos.org)

## External resources
- [LunarG Vulkan SDK](https://vulkan.lunarg.com)
- [Official list of Vulkan resources](https://www.khronos.org/vulkan/resources)
- [Vulkan API specifications](https://www.khronos.org/registry/vulkan/specs/1.0/apispec.html) ([quick reference cards](https://www.khronos.org/registry/vulkan/specs/1.0/refguide/Vulkan-1.0-web.pdf))
- [SPIR-V specifications](https://www.khronos.org/registry/spir-v/specs/1.0/SPIRV.html)
- [My personal view on Vulkan (as a hobby developer)](http://www.saschawillems.de/?p=1886)
