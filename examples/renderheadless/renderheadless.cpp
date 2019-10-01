/*
* Vulkan Example - Minimal headless rendering example
*
* Copyright (C) 2017 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include <common.hpp>
#include <utils.hpp>
#include <vks/debug.hpp>
#include <vks/context.hpp>
#include <vks/pipelines.hpp>
//#include <vks/helpers.hpp>

#include "matrix.h"

#define BUFFER_ELEMENTS 32

#if defined(VK_USE_PLATFORM_ANDROID_KHR)
#define LOG(...) ((void)__android_log_print(ANDROID_LOG_INFO, "vulkanExample", __VA_ARGS__))
#else
#define LOG(...) printf(__VA_ARGS__)
#endif

struct ShareHandles {
    HANDLE memory{ INVALID_HANDLE_VALUE };
    HANDLE glReady{ INVALID_HANDLE_VALUE };
    HANDLE glComplete{ INVALID_HANDLE_VALUE };
};

class VulkanExample {
public:
    struct SharedResources {
        vks::Image texture;
        struct {
            vk::Semaphore glReady;
            vk::Semaphore glComplete;
        } semaphores;
        vk::CommandBuffer transitionCmdBuf;
        ShareHandles handles;
        vk::Device device;

        void init(const vks::Context& context) {
            device = context.device;
            vk::DispatchLoaderDynamic dynamicLoader{ context.instance, &vkGetInstanceProcAddr, device, &vkGetDeviceProcAddr };
            {
                auto handleType = vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueWin32;
                {
                    vk::SemaphoreCreateInfo sci;
                    vk::ExportSemaphoreCreateInfo esci;
                    sci.pNext = &esci;
                    esci.handleTypes = handleType;
                    semaphores.glReady = device.createSemaphore(sci);
                    semaphores.glComplete = device.createSemaphore(sci);
                }
                handles.glReady = device.getSemaphoreWin32HandleKHR({ semaphores.glReady, handleType }, dynamicLoader);
                handles.glComplete = device.getSemaphoreWin32HandleKHR({ semaphores.glComplete, handleType }, dynamicLoader);
            }

            {
                vk::ImageCreateInfo imageCreateInfo;
                imageCreateInfo.imageType = vk::ImageType::e2D;
                imageCreateInfo.format = vk::Format::eR8G8B8A8Unorm;
                imageCreateInfo.mipLevels = 1;
                imageCreateInfo.arrayLayers = 1;
                imageCreateInfo.extent.depth = 1;
                imageCreateInfo.extent.width = 1024;
                imageCreateInfo.extent.height = 1024;
                imageCreateInfo.initialLayout = vk::ImageLayout::eUndefined;
                imageCreateInfo.usage = vk::ImageUsageFlagBits::eColorAttachment;  // | vk::ImageUsageFlagBits::eSampled;
                texture.image = device.createImage(imageCreateInfo);
                texture.device = device;
                texture.format = imageCreateInfo.format;
                texture.extent = imageCreateInfo.extent;
            }

            {
                vk::MemoryRequirements memReqs = device.getImageMemoryRequirements(texture.image);
                vk::MemoryAllocateInfo memAllocInfo;
                vk::ExportMemoryAllocateInfo exportAllocInfo{ vk::ExternalMemoryHandleTypeFlagBits::eOpaqueWin32 };
                memAllocInfo.pNext = &exportAllocInfo;
                memAllocInfo.allocationSize = texture.allocSize = memReqs.size;
                memAllocInfo.memoryTypeIndex = context.getMemoryType(memReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
                texture.memory = device.allocateMemory(memAllocInfo);
                device.bindImageMemory(texture.image, texture.memory, 0);
                handles.memory = device.getMemoryWin32HandleKHR({ texture.memory, vk::ExternalMemoryHandleTypeFlagBits::eOpaqueWin32 }, dynamicLoader);
            }

            {
                // Create sampler
                vk::SamplerCreateInfo samplerCreateInfo;
                samplerCreateInfo.magFilter = vk::Filter::eLinear;
                samplerCreateInfo.minFilter = vk::Filter::eLinear;
                samplerCreateInfo.mipmapMode = vk::SamplerMipmapMode::eLinear;
                // Max level-of-detail should match mip level count
                samplerCreateInfo.maxLod = (float)1;
                // Only enable anisotropic filtering if enabled on the devicec
                samplerCreateInfo.maxAnisotropy = context.deviceFeatures.samplerAnisotropy ? context.deviceProperties.limits.maxSamplerAnisotropy : 1.0f;
                samplerCreateInfo.anisotropyEnable = VK_FALSE;  //context.deviceFeatures.samplerAnisotropy;
                samplerCreateInfo.borderColor = vk::BorderColor::eFloatOpaqueWhite;
                texture.sampler = device.createSampler(samplerCreateInfo);
            }
        }

        void destroy() {
            texture.destroy();
            device.destroy(semaphores.glComplete);
            device.destroy(semaphores.glReady);
        }

        void transitionToGl(const vk::Queue& queue) const {
            vk::SubmitInfo submitInfo;
            vk::PipelineStageFlags stageFlags = vk::PipelineStageFlagBits::eBottomOfPipe;
            submitInfo.pWaitDstStageMask = &stageFlags;
            submitInfo.waitSemaphoreCount = 0;
            submitInfo.signalSemaphoreCount = 1;
            submitInfo.pSignalSemaphores = &semaphores.glReady;

            // Command buffer count is 0 because we have already transitioned the image!
            // This call is just to tell GL about it.
            submitInfo.commandBufferCount = 0;
            // Interestingly, none of this matters if submitInfo.commandBufferCount == 0
            submitInfo.pCommandBuffers = &transitionCmdBuf;
            queue.submit({ submitInfo }, {});
        }
    } m_sharedResource;
    vks::Context context;
    vk::Instance instance;
    vk::Device& device{ context.device };
    vk::DescriptorSetLayout descriptorSetLayout;
    vk::PipelineLayout pipelineLayout;
    vk::Pipeline pipeline;
    vks::Buffer vertexBuffer, indexBuffer;

    vk::Extent2D size;
    uint32_t& width{ size.width };
    uint32_t& height{ size.height };
    vk::RenderPass renderPass;
    vk::Framebuffer framebuffer;
    // Color attachment is now the shared resource.
    vks::Image depthAttachment;

    /*
        Submit command buffer to a queue and wait for fence until queue operations have been finished
    */
    void submitWork(const vk::CommandBuffer& cmdBuffer, const vk::Queue queue, const vk::Semaphore& waitSemaphore) {
        vk::SubmitInfo submitInfo;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &cmdBuffer;

        vk::PipelineStageFlags stageFlags = vk::PipelineStageFlagBits::eBottomOfPipe;
        submitInfo.pWaitDstStageMask = &stageFlags;

        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = &m_sharedResource.semaphores.glComplete;

        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = &m_sharedResource.semaphores.glReady;

        // This would never have been correct because we create the
        // fence but don't pass it to the submit call.
        //vk::Fence fence = device.createFence({});
        queue.submit({ submitInfo }, {});
        //device.waitForFences(fence, true, UINT64_MAX);
        //device.destroy(fence);
    }

    void doRendering(int time) {
        vk::CommandBuffer commandBuffer = context.allocateCommandBuffers(1)[0];
        commandBuffer.begin(vk::CommandBufferBeginInfo{});

        vk::ClearValue clearValues[2];
        clearValues[0].color = vks::util::clearColor({ 0.0f, 0.0f, 0.2f, 1.0f });
        clearValues[1].depthStencil = vk::ClearDepthStencilValue{ 1.0f, 0 };

        vk::RenderPassBeginInfo renderPassBeginInfo{ renderPass, framebuffer, vk::Rect2D{ vk::Offset2D{}, size }, 2, clearValues };
        commandBuffer.beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);

        vk::Viewport viewport = {};
        viewport.height = (float)height;
        viewport.width = (float)width;
        viewport.minDepth = (float)0.0f;
        viewport.maxDepth = (float)1.0f;
        commandBuffer.setViewport(0, viewport);

        // Update dynamic scissor state
        vk::Rect2D scissor;
        scissor.extent = size;
        commandBuffer.setScissor(0, scissor);
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);

        // Render scene
        vk::DeviceSize offset = 0;
        commandBuffer.bindVertexBuffers(0, vertexBuffer.buffer, offset);
        commandBuffer.bindIndexBuffer(indexBuffer.buffer, offset, vk::IndexType::eUint32);

        std::vector<glm::vec3> pos = {
            glm::vec3(-1.5f, 0.0f, -4.0f),
            glm::vec3(0.0f, 0.0f, -2.5f),
            glm::vec3(1.5f, 0.0f, -4.0f),
        };

        for (auto v : pos) {
            // TODO: add a rotation matrix here to show that the image is being continuously rendered by Vulkan.
            // Rotation code here:https://vulkan-tutorial.com/Uniform_buffers/Descriptor_layout_and_buffer
            // Actually, this is much easier...
            glm::mat4 mvpMatrix = glm::perspective(glm::radians(60.0f + (float(time % 120))), (float)width / (float)height, 0.1f, 256.0f) * glm::translate(glm::mat4(1.0f), v);
            commandBuffer.pushConstants<glm::mat4>(pipelineLayout, vk::ShaderStageFlagBits::eVertex, 0, mvpMatrix);
            commandBuffer.drawIndexed(3, 1, 0, 0, 0);
        }

        commandBuffer.endRenderPass();
        commandBuffer.end();

        submitWork(commandBuffer, context.queue, m_sharedResource.semaphores.glComplete);
        // This call was used only to signal the glReady semaphore, which is
        // now done in submitWork() above.
        //m_sharedResource.transitionToGl(context.queue);
    }

    VulkanExample() {
        LOG("Running headless rendering example\n");

#if defined(VK_USE_PLATFORM_ANDROID_KHR)
        LOG("loading vulkan lib");
        vks::android::loadVulkanLibrary();
#endif

        vk::ApplicationInfo appInfo;
        appInfo.pApplicationName = "Vulkan headless example";
        appInfo.pEngineName = "VulkanExample";
        appInfo.apiVersion = VK_MAKE_VERSION(1, 0, 0);

        /*
            Vulkan instance creation (without surface extensions)
        */
        context.requireExtensions({
            VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME,    //
            VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME  //
        });

        context.requireDeviceExtensions({
            VK_KHR_MAINTENANCE1_EXTENSION_NAME,            //
                VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,     //
                VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,  //
#if defined(WIN32)
                VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME,    //
                VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME  //
#else
                VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,    //
                VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME  //
#endif
        });

        vk::InstanceCreateInfo instanceCreateInfo;
        instanceCreateInfo.pApplicationInfo = &appInfo;

#if DEBUG
        context.setValidationEnabled(true);
#endif
        context.createInstance();
        context.createDevice();

#if defined(VK_USE_PLATFORM_ANDROID_KHR)
        vks::android::loadVulkanFunctions(instance);
#endif

        /*
            Prepare vertex and index buffers
        */
        struct Vertex {
            float position[3];
            float color[3];
        };

        {
            std::vector<Vertex> vertices = { { { 1.0f, 1.0f, 0.0f }, { 1.0f, 0.0f, 0.0f } },
                                             { { -1.0f, 1.0f, 0.0f }, { 0.0f, 1.0f, 0.0f } },
                                             { { 0.0f, -1.0f, 0.0f }, { 0.0f, 0.0f, 1.0f } } };
            std::vector<uint32_t> indices = { 0, 1, 2 };

            // Vertices
            vertexBuffer = context.stageToDeviceBuffer(vk::BufferUsageFlagBits::eVertexBuffer, vertices);

            // Indices
            indexBuffer = context.stageToDeviceBuffer(vk::BufferUsageFlagBits::eIndexBuffer, indices);
        }

        /*
            Create framebuffer attachments
        */
        width = 1024;
        height = 1024;
        static const vk::Format colorFormat = vk::Format::eR8G8B8A8Unorm;
        static const vk::Format depthFormat = context.getSupportedDepthFormat();

        // Initialize the shared resource.
        m_sharedResource.init(context);
        vks::Image& colorAttachment = m_sharedResource.texture;
        {
            vk::ImageViewCreateInfo imageView;
            imageView.viewType = vk::ImageViewType::e2D;
            imageView.format = colorFormat;
            imageView.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
            imageView.subresourceRange.baseMipLevel = 0;
            imageView.subresourceRange.levelCount = 1;
            imageView.subresourceRange.baseArrayLayer = 0;
            imageView.subresourceRange.layerCount = 1;
            imageView.image = colorAttachment.image;
            colorAttachment.view = device.createImageView(imageView);

            vk::ImageCreateInfo image;
            image.imageType = vk::ImageType::e2D;
            image.format = colorFormat;
            image.extent.width = width;
            image.extent.height = height;
            image.extent.depth = 1;
            image.mipLevels = 1;
            image.arrayLayers = 1;
            // Depth stencil attachment
            image.format = depthFormat;
            image.usage = vk::ImageUsageFlagBits::eDepthStencilAttachment;
            depthAttachment = context.createImage(image);

            imageView.format = depthFormat;
            imageView.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil;
            imageView.image = depthAttachment.image;
            depthAttachment.view = device.createImageView(imageView);
        }

        /*
            Create renderpass
        */
        {
            std::array<vk::AttachmentDescription, 2> attchmentDescriptions = {};
            // Color attachment
            attchmentDescriptions[0].format = colorFormat;
            attchmentDescriptions[0].loadOp = vk::AttachmentLoadOp::eClear;
            attchmentDescriptions[0].storeOp = vk::AttachmentStoreOp::eStore;
            attchmentDescriptions[0].stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
            attchmentDescriptions[0].stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
            attchmentDescriptions[0].initialLayout = vk::ImageLayout::eUndefined;
            attchmentDescriptions[0].finalLayout = vk::ImageLayout::eColorAttachmentOptimal;
            // Depth attachment
            attchmentDescriptions[1].format = depthFormat;
            attchmentDescriptions[1].loadOp = vk::AttachmentLoadOp::eClear;
            attchmentDescriptions[1].storeOp = vk::AttachmentStoreOp::eDontCare;
            attchmentDescriptions[1].stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
            attchmentDescriptions[1].stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
            attchmentDescriptions[1].initialLayout = vk::ImageLayout::eUndefined;
            attchmentDescriptions[1].finalLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal;

            vk::AttachmentReference colorReference = { 0, vk::ImageLayout::eColorAttachmentOptimal };
            vk::AttachmentReference depthReference = { 1, vk::ImageLayout::eDepthStencilAttachmentOptimal };

            vk::SubpassDescription subpassDescription;
            subpassDescription.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
            subpassDescription.colorAttachmentCount = 1;
            subpassDescription.pColorAttachments = &colorReference;
            subpassDescription.pDepthStencilAttachment = &depthReference;

            // Use subpass dependencies for layout transitions
            std::array<vk::SubpassDependency, 2> dependencies;

            dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
            dependencies[0].dstSubpass = 0;
            dependencies[0].srcStageMask = vk::PipelineStageFlagBits::eBottomOfPipe;
            dependencies[0].dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
            dependencies[0].srcAccessMask = vk::AccessFlagBits::eMemoryRead;
            dependencies[0].dstAccessMask = vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite;
            dependencies[0].dependencyFlags = vk::DependencyFlagBits::eByRegion;

            dependencies[1].srcSubpass = 0;
            dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
            dependencies[1].srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
            dependencies[1].dstStageMask = vk::PipelineStageFlagBits::eBottomOfPipe;
            dependencies[1].srcAccessMask = vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite;
            dependencies[1].dstAccessMask = vk::AccessFlagBits::eMemoryRead;
            dependencies[1].dependencyFlags = vk::DependencyFlagBits::eByRegion;

            // Create the actual renderpass
            vk::RenderPassCreateInfo renderPassInfo;
            renderPassInfo.attachmentCount = static_cast<uint32_t>(attchmentDescriptions.size());
            renderPassInfo.pAttachments = attchmentDescriptions.data();
            renderPassInfo.subpassCount = 1;
            renderPassInfo.pSubpasses = &subpassDescription;
            renderPassInfo.dependencyCount = static_cast<uint32_t>(dependencies.size());
            renderPassInfo.pDependencies = dependencies.data();
            renderPass = device.createRenderPass(renderPassInfo);

            vk::ImageView attachments[2];
            attachments[0] = colorAttachment.view;
            attachments[1] = depthAttachment.view;

            vk::FramebufferCreateInfo framebufferCreateInfo;
            framebufferCreateInfo.renderPass = renderPass;
            framebufferCreateInfo.attachmentCount = 2;
            framebufferCreateInfo.pAttachments = attachments;
            framebufferCreateInfo.width = width;
            framebufferCreateInfo.height = height;
            framebufferCreateInfo.layers = 1;
            framebuffer = device.createFramebuffer(framebufferCreateInfo);
        }

        /* 
            Prepare graphics pipeline
        */
        {
            descriptorSetLayout = device.createDescriptorSetLayout({});

            vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo;
            // MVP via push constant block
            vk::PushConstantRange pushConstantRange{ vk::ShaderStageFlagBits::eVertex, 0, sizeof(glm::mat4) };
            pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
            pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;
            pipelineLayout = device.createPipelineLayout(pipelineLayoutCreateInfo);

            // Create pipeline
            vks::pipelines::GraphicsPipelineBuilder builder{ device, pipelineLayout, renderPass };
            builder.rasterizationState.frontFace = vk::FrontFace::eClockwise;

            // Vertex bindings an attributes
            // Binding description
            builder.vertexInputState.bindingDescriptions = {
                vk::VertexInputBindingDescription{ 0, sizeof(Vertex), vk::VertexInputRate::eVertex },
            };

            // Attribute descriptions
            builder.vertexInputState.attributeDescriptions = {
                vk::VertexInputAttributeDescription{ 0, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, position) },  // Position
                vk::VertexInputAttributeDescription{ 1, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, color) },     // Color
            };

            builder.loadShader(vkx::getAssetPath() + "shaders/renderheadless/triangle.vert.spv", vk::ShaderStageFlagBits::eVertex);
            builder.loadShader(vkx::getAssetPath() + "shaders/renderheadless/triangle.frag.spv", vk::ShaderStageFlagBits::eFragment);
            pipeline = builder.create(context.pipelineCache);
        }
    }

    ~VulkanExample() {
        m_sharedResource.destroy();
        vertexBuffer.destroy();
        indexBuffer.destroy();
        depthAttachment.destroy();
        device.destroy(renderPass);
        device.destroy(framebuffer);
        device.destroy(pipelineLayout);
        device.destroy(descriptorSetLayout);
        device.destroy(pipeline);

        context.destroy();
    }
};

#if defined(VK_USE_PLATFORM_ANDROID_KHR)
void handleAppCommand(android_app* app, int32_t cmd) {
    if (cmd == APP_CMD_INIT_WINDOW) {
        VulkanExample* vulkanExample = new VulkanExample();
        delete (vulkanExample);
        ANativeActivity_finish(app->activity);
    }
}
void android_main(android_app* state) {
    app_dummy();
    androidapp = state;
    androidapp->onAppCmd = handleAppCommand;
    int ident, events;
    struct android_poll_source* source;
    while ((ident = ALooper_pollAll(-1, NULL, &events, (void**)&source)) >= 0) {
        if (source != NULL) {
            source->process(androidapp, source);
        }
        if (androidapp->destroyRequested != 0) {
            break;
        }
    }
}
#else

////////////////minimalOpenGL.h////////////////

#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cassert>
#include <vector>

void APIENTRY debugCallback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, const void* userParam) {
    if ((type == GL_DEBUG_TYPE_ERROR) || (type == GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR)) {
        fprintf(stderr, "GL Debug: %s\n", message);
    }
}

GLFWwindow* initOpenGL(int width, int height, bool fullscreen, bool vsync, const std::string& title) {
    if (!glfwInit()) {
        fprintf(stderr, "ERROR: could not start GLFW\n");
        ::exit(1);
    }

    // Without these, shaders actually won't initialize properly
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
    glfwWindowHint(GLFW_DECORATED, fullscreen ? GL_FALSE : GL_TRUE);
    // Highest
    //    glfwWindowHint(GLFW_REFRESH_RATE, GLFW_DONT_CARE);
    glfwWindowHint(GLFW_REFRESH_RATE, 144);

#ifdef _DEBUG
    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GL_TRUE);
#endif

    GLFWwindow* window = glfwCreateWindow(width, height, title.c_str(), fullscreen ? glfwGetPrimaryMonitor() : nullptr, nullptr);
    if (!window) {
        fprintf(stderr, "ERROR: could not open window with GLFW\n");
        glfwTerminate();
        ::exit(2);
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(vsync ? 1 : 0);

    // Start GLEW extension handler, with improved support for new features
    glewExperimental = GL_TRUE;
    glewInit();

    // Clear startup errors
    while (glGetError() != GL_NONE) {
    }

#ifdef _DEBUG
    glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
    glEnable(GL_DEBUG_OUTPUT);
#ifndef _OSX
    // Causes a segmentation fault on OS X
    glDebugMessageCallback(debugCallback, nullptr);
#endif
#endif

    fprintf(stderr, "GPU: %s (OpenGL version %s)\n", glGetString(GL_RENDERER), glGetString(GL_VERSION));

    // Bind a single global vertex array (done this way since OpenGL 3)
    {
        GLuint vao;
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
    }

    // Check for errors
    {
        const GLenum error = glGetError();
        assert(error == GL_NONE);
    }

    return window;
}

std::string loadTextFile(const std::string& filename) {
    std::stringstream buffer;
    buffer << std::ifstream(filename.c_str()).rdbuf();
    return buffer.str();
}

GLuint compileShaderStage(GLenum stage, const std::string& source) {
    GLuint shader = glCreateShader(stage);
    const char* srcArray[] = { source.c_str() };

    glShaderSource(shader, 1, srcArray, NULL);
    glCompileShader(shader);

    GLint success = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);

    if (success == GL_FALSE) {
        GLint logSize = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &logSize);

        std::vector<GLchar> errorLog(logSize);
        glGetShaderInfoLog(shader, logSize, &logSize, &errorLog[0]);

        fprintf(stderr, "Error while compiling\n %s\n\nError: %s\n", source.c_str(), &errorLog[0]);
        assert(false);

        glDeleteShader(shader);
        shader = GL_NONE;
    }

    return shader;
}

GLuint createShaderProgram(const std::string& vertexShaderSource, const std::string& pixelShaderSource) {
    GLuint shader = glCreateProgram();

    glAttachShader(shader, compileShaderStage(GL_VERTEX_SHADER, vertexShaderSource));
    glAttachShader(shader, compileShaderStage(GL_FRAGMENT_SHADER, pixelShaderSource));
    glLinkProgram(shader);

    return shader;
}

/** Submits a full-screen quad at the far plane and runs a procedural sky shader on it.

    All matrices are 4x4 row-major
    \param light Light vector, must be normalized 
 */
void drawSky(int windowWidth,
             int windowHeight,
             float nearPlaneZ,
             float farPlaneZ,
             const float* cameraToWorldMatrix,
             const float* projectionMatrixInverse,
             const float* light) {
#define VERTEX_SHADER(s) "#version 410\n" #s
#define PIXEL_SHADER(s) VERTEX_SHADER(s)

    static const GLuint skyShader =
        createShaderProgram(VERTEX_SHADER(void main() { gl_Position = vec4(gl_VertexID & 1, gl_VertexID >> 1, 0.0, 0.5) * 4.0 - 1.0; }),

                            PIXEL_SHADER(out vec3 pixelColor;

                                         uniform vec3 light; uniform vec2 resolution; uniform mat4 cameraToWorldMatrix; uniform mat4 invProjectionMatrix;

                                         float hash(vec2 p) { return fract(1e4 * sin(17.0 * p.x + p.y * 0.1) * (0.1 + abs(sin(p.y * 13.0 + p.x)))); }

                                         float noise(vec2 x) {
                                             vec2 i = floor(x);
                                             float a = hash(i);
                                             float b = hash(i + vec2(1.0, 0.0));
                                             float c = hash(i + vec2(0.0, 1.0));
                                             float d = hash(i + vec2(1.0, 1.0));

                                             vec2 f = fract(x);
                                             vec2 u = f * f * (3.0 - 2.0 * f);
                                             return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
                                         }

                                         float fbm(vec2 p) {
                                             const mat2 m2 = mat2(0.8, -0.6, 0.6, 0.8);
                                             float f = 0.5000 * noise(p);
                                             p = m2 * p * 2.02;
                                             f += 0.2500 * noise(p);
                                             p = m2 * p * 2.03;
                                             f += 0.1250 * noise(p);
                                             p = m2 * p * 2.01;
                                             f += 0.0625 * noise(p);
                                             return f / 0.9375;
                                         }

                                         vec3 render(in vec3 sun, in vec3 ro, in vec3 rd, in float resolution) {
                                             vec3 col;
                                             if (rd.y < 0.0) {
                                                 // Ground
                                                 float t = -ro.y / rd.y;
                                                 vec2 P = ro.xz + t * rd.xz + vec2(0.5);
                                                 vec2 Q = floor(P);
                                                 // 1m^2 grid
                                                 P = mod(P, 1.0);

                                                 const float gridLineWidth = 0.1;
                                                 float res = clamp(3000.0 / resolution, 1.0, 4.0);
                                                 P = 1.0 - abs(P - 0.5) * 2.0;
                                                 float d = clamp(min(P.x, P.y) / (gridLineWidth * clamp(t + res * 2.0, 1.0, 3.0)) + 0.5, 0.0, 1.0);
                                                 float shade = mix(hash(100.0 + Q * 0.1) * 0.4, 0.3, min(t * t * 0.00001 / max(-rd.y, 0.001), 1.0)) + 0.6;
                                                 col = vec3(pow(d, clamp(150.0 / (pow(max(t - 2.0, 0.1), res) + 1.0), 0.1, 15.0))) * shade + 0.1;
                                             } else {
                                                 // Sky
                                                 col = vec3(0.3, 0.55, 0.8) * (1.0 - 0.8 * rd.y) * 0.9;
                                                 float sundot = clamp(dot(rd, sun), 0.0, 1.0);
                                                 col += 0.25 * vec3(1.0, 0.7, 0.4) * pow(sundot, 8.0);
                                                 col += 0.75 * vec3(1.0, 0.8, 0.5) * pow(sundot, 64.0);
                                                 col = mix(col, vec3(1.0, 0.95, 1.0),
                                                           0.5 * smoothstep(0.5, 0.8, fbm((ro.xz + rd.xz * (250000.0 - ro.y) / rd.y) * 0.000008)));
                                             }
                                             return mix(col, vec3(0.7, 0.75, 0.8), pow(1.0 - max(abs(rd.y), 0.0), 8.0));
                                         }

                                         void main() {
                                             vec3 rd =
                                                 normalize(mat3(cameraToWorldMatrix) *
                                                           vec3((invProjectionMatrix * vec4(gl_FragCoord.xy / resolution.xy * 2.0 - 1.0, -1.0, 1.0)).xy, -1.0));
                                             pixelColor = render(light, cameraToWorldMatrix[3].xyz, rd, resolution.x);
                                         }));

    static const GLint lightUniform = glGetUniformLocation(skyShader, "light");
    static const GLint resolutionUniform = glGetUniformLocation(skyShader, "resolution");
    static const GLint cameraToWorldMatrixUniform = glGetUniformLocation(skyShader, "cameraToWorldMatrix");
    static const GLint invProjectionMatrixUniform = glGetUniformLocation(skyShader, "invProjectionMatrix");

    glDisable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);
    glDisable(GL_CULL_FACE);

    glUseProgram(skyShader);
    glUniform3fv(lightUniform, 1, light);
    glUniform2f(resolutionUniform, float(windowWidth), float(windowHeight));
    glUniformMatrix4fv(cameraToWorldMatrixUniform, 1, GL_TRUE, cameraToWorldMatrix);
    glUniformMatrix4fv(invProjectionMatrixUniform, 1, GL_TRUE, projectionMatrixInverse);
    glDrawArrays(GL_TRIANGLES, 0, 3);

#undef PIXEL_SHADER
#undef VERTEX_SHADER
}

namespace Cube {
const float position[][3] = { -.5f, .5f,  -.5f, -.5f, .5f,  .5f,  .5f,  .5f,  .5f,  .5f,  .5f,  -.5f, -.5f, .5f,  -.5f, -.5f, -.5f, -.5f,
                              -.5f, -.5f, .5f,  -.5f, .5f,  .5f,  .5f,  .5f,  .5f,  .5f,  -.5f, .5f,  .5f,  -.5f, -.5f, .5f,  .5f,  -.5f,
                              .5f,  .5f,  -.5f, .5f,  -.5f, -.5f, -.5f, -.5f, -.5f, -.5f, .5f,  -.5f, -.5f, .5f,  .5f,  -.5f, -.5f, .5f,
                              .5f,  -.5f, .5f,  .5f,  .5f,  .5f,  -.5f, -.5f, .5f,  -.5f, -.5f, -.5f, .5f,  -.5f, -.5f, .5f,  -.5f, .5f };
const float normal[][3] = { 0.f, 1.f, 0.f, 0.f, 1.f, 0.f, 0.f, 1.f, 0.f, 0.f, 1.f, 0.f, -1.f, 0.f,  0.f,  -1.f, 0.f,  0.f,  -1.f, 0.f,  0.f,  -1.f, 0.f,  0.f,
                            1.f, 0.f, 0.f, 1.f, 0.f, 0.f, 1.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f,  0.f,  -1.f, 0.f,  0.f,  -1.f, 0.f,  0.f,  -1.f, 0.f,  0.f,  -1.f,
                            0.f, 0.f, 1.f, 0.f, 0.f, 1.f, 0.f, 0.f, 1.f, 0.f, 0.f, 1.f, 0.f,  -1.f, 0.f,  0.f,  -1.f, 0.f,  0.f,  -1.f, 0.f,  0.f,  -1.f, 0.f };
const float tangent[][4] = { 1.f,  0.f, 0.f, 1.f, 1.f,  0.f, 0.f, 1.f, 1.f,  0.f, 0.f,  1.f, 1.f,  0.f, 0.f,  1.f, 0.f, 0.f, 1.f,  1.f, 0.f, 0.f, 1.f,  1.f,
                             0.f,  0.f, 1.f, 1.f, 0.f,  0.f, 1.f, 1.f, 0.f,  0.f, -1.f, 1.f, 0.f,  0.f, -1.f, 1.f, 0.f, 0.f, -1.f, 1.f, 0.f, 0.f, -1.f, 1.f,
                             -1.f, 0.f, 0.f, 1.f, -1.f, 0.f, 0.f, 1.f, -1.f, 0.f, 0.f,  1.f, -1.f, 0.f, 0.f,  1.f, 1.f, 0.f, 0.f,  1.f, 1.f, 0.f, 0.f,  1.f,
                             1.f,  0.f, 0.f, 1.f, 1.f,  0.f, 0.f, 1.f, 1.f,  0.f, 0.f,  1.f, 1.f,  0.f, 0.f,  1.f, 1.f, 0.f, 0.f,  1.f, 1.f, 0.f, 0.f,  1.f };
const float texCoord[][2] = { 0.f, 0.f, 0.f, 1.f, 1.f, 1.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 1.f, 1.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 1.f, 1.f, 1.f, 0.f,
                              0.f, 0.f, 0.f, 1.f, 1.f, 1.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 1.f, 1.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 1.f, 1.f, 1.f, 0.f };
const int index[] = { 0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7, 8, 9, 10, 8, 10, 11, 12, 13, 14, 12, 14, 15, 16, 17, 18, 16, 18, 19, 20, 21, 22, 20, 22, 23 };
const float scale = 1.0f;
};  // namespace Cube

/** Loads a 24- or 32-bit BMP file into memory */
void loadBMP(const std::string& filename, int& width, int& height, int& channels, std::vector<std::uint8_t>& data) {
    std::fstream hFile(filename.c_str(), std::ios::in | std::ios::binary);
    if (!hFile.is_open()) {
        throw std::invalid_argument("Error: File Not Found.");
    }

    hFile.seekg(0, std::ios::end);
    size_t len = hFile.tellg();
    hFile.seekg(0, std::ios::beg);
    std::vector<std::uint8_t> header(len);
    hFile.read(reinterpret_cast<char*>(header.data()), 54);

    if ((header[0] != 'B') && (header[1] != 'M')) {
        hFile.close();
        throw std::invalid_argument("Error: File is not a BMP.");
    }

    if ((header[28] != 24) && (header[28] != 32)) {
        hFile.close();
        throw std::invalid_argument("Error: File is not uncompressed 24 or 32 bits per pixel.");
    }

    const short bitsPerPixel = header[28];
    channels = bitsPerPixel / 8;
    width = header[18] + (header[19] << 8);
    height = header[22] + (header[23] << 8);
    std::uint32_t offset = header[10] + (header[11] << 8);
    std::uint32_t size = ((width * bitsPerPixel + 31) / 32) * 4 * height;
    data.resize(size);

    hFile.seekg(offset, std::ios::beg);
    hFile.read(reinterpret_cast<char*>(data.data()), size);
    hFile.close();

    // Flip the y axis
    std::vector<std::uint8_t> tmp;
    const size_t rowBytes = width * channels;
    tmp.resize(rowBytes);
    for (int i = height / 2 - 1; i >= 0; --i) {
        const int j = height - 1 - i;
        // Swap the rows
        memcpy(tmp.data(), &data[i * rowBytes], rowBytes);
        memcpy(&data[i * rowBytes], &data[j * rowBytes], rowBytes);
        memcpy(&data[j * rowBytes], tmp.data(), rowBytes);
    }

    // Convert BGR[A] format to RGB[A] format
    if (channels == 4) {
        // BGRA
        std::uint32_t* p = reinterpret_cast<std::uint32_t*>(data.data());
        for (int i = width * height - 1; i >= 0; --i) {
            const unsigned int x = p[i];
            p[i] = ((x >> 24) & 0xFF) | (((x >> 16) & 0xFF) << 8) | (((x >> 8) & 0xFF) << 16) | ((x & 0xFF) << 24);
        }
    } else {
        // BGR
        for (int i = (width * height - 1) * 3; i >= 0; i -= 3) {
            std::swap(data[i], data[i + 2]);
        }
    }
}

#ifndef _MSC_VER
#pragma clang diagnostic pop
#endif

GLFWwindow* window = nullptr;

#ifndef Shape
#define Shape Cube
#endif

int main() {
    VulkanExample* vulkanExample = new VulkanExample();
    std::cout << "Finished. Proceeding to GL...\n";

    std::cout << "Minimal OpenGL 4.1 Example by Morgan McGuire\n\nW, A, S, D, C, Z keys to translate\nMouse click and drag to rotate\nESC to quit\n\n";
    std::cout << std::fixed;
    const bool fullScreen = false;
    bool vsync = true;

    uint32_t framebufferWidth = 1280, framebufferHeight = 720;
#ifdef _VR
    const int numEyes = 2;
    hmd = initOpenVR(framebufferWidth, framebufferHeight);
    vsync = false;
    assert(hmd);
#else
    const int numEyes = 1;
#endif

    int windowHeight = 720;
    int windowWidth = (framebufferWidth * windowHeight) / framebufferHeight;
    if (fullScreen) {
        // Override settings above
        windowWidth = framebufferWidth = 1920;
        windowHeight = framebufferHeight = 1080;
    }

    window = initOpenGL(windowWidth, windowHeight, fullScreen, vsync, "minimalVKGLInterop");

    Vector3 bodyTranslation(0.0f, 1.6f, 5.0f);
    Vector3 bodyRotation;

    //////////////////////////////////////////////////////////////////////
    // Allocate the frame buffer. This code allocates one framebuffer per eye.
    // That requires more GPU memory, but is useful when performing temporal
    // filtering or making render calls that can target both simultaneously.

    GLuint framebuffer[numEyes];
    glGenFramebuffers(numEyes, framebuffer);

    GLuint colorRenderTarget[numEyes], depthRenderTarget[numEyes];
    glGenTextures(numEyes, colorRenderTarget);
    glGenTextures(numEyes, depthRenderTarget);
    for (int eye = 0; eye < numEyes; ++eye) {
        glBindTexture(GL_TEXTURE_2D, colorRenderTarget[eye]);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, framebufferWidth, framebufferHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

        glBindTexture(GL_TEXTURE_2D, depthRenderTarget[eye]);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, framebufferWidth, framebufferHeight, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, nullptr);

        glBindFramebuffer(GL_FRAMEBUFFER, framebuffer[eye]);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, colorRenderTarget[eye], 0);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthRenderTarget[eye], 0);
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    /////////////////////////////////////////////////////////////////
    // Load vertex array buffers

    GLuint positionBuffer = GL_NONE;
    glGenBuffers(1, &positionBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, positionBuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(Shape::position), Shape::position, GL_STATIC_DRAW);

    GLuint texCoordBuffer = GL_NONE;
    glGenBuffers(1, &texCoordBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, texCoordBuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(Shape::texCoord), Shape::texCoord, GL_STATIC_DRAW);

    GLuint normalBuffer = GL_NONE;
    glGenBuffers(1, &normalBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, normalBuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(Shape::normal), Shape::normal, GL_STATIC_DRAW);

    GLuint tangentBuffer = GL_NONE;
    glGenBuffers(1, &tangentBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, tangentBuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(Shape::tangent), Shape::tangent, GL_STATIC_DRAW);

    const int numVertices = sizeof(Shape::position) / sizeof(Shape::position[0]);
    (void)numVertices;

    GLuint indexBuffer = GL_NONE;
    glGenBuffers(1, &indexBuffer);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBuffer);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(Shape::index), Shape::index, GL_STATIC_DRAW);
    const int numIndices = sizeof(Shape::index) / sizeof(Shape::index[0]);

    /////////////////////////////////////////////////////////////////////
    // Create the main shader
    const GLuint shader = createShaderProgram(loadTextFile("min.vrt"), loadTextFile("min.pix"));

    // Binding points for attributes and uniforms discovered from the shader
    const GLint positionAttribute = glGetAttribLocation(shader, "position");
    const GLint normalAttribute = glGetAttribLocation(shader, "normal");
    const GLint texCoordAttribute = glGetAttribLocation(shader, "texCoord");
    const GLint tangentAttribute = glGetAttribLocation(shader, "tangent");
    const GLint colorTextureUniform = glGetUniformLocation(shader, "colorTexture");

    const GLuint uniformBlockIndex = glGetUniformBlockIndex(shader, "Uniform");
    const GLuint uniformBindingPoint = 1;
    glUniformBlockBinding(shader, uniformBlockIndex, uniformBindingPoint);

    GLuint uniformBlock;
    glGenBuffers(1, &uniformBlock);

    {
        // Allocate space for the uniform block buffer
        GLint uniformBlockSize;
        glGetActiveUniformBlockiv(shader, uniformBlockIndex, GL_UNIFORM_BLOCK_DATA_SIZE, &uniformBlockSize);
        glBindBuffer(GL_UNIFORM_BUFFER, uniformBlock);
        glBufferData(GL_UNIFORM_BUFFER, uniformBlockSize, nullptr, GL_DYNAMIC_DRAW);
    }

    const GLchar* uniformName[] = { "Uniform.objectToWorldNormalMatrix", "Uniform.objectToWorldMatrix", "Uniform.modelViewProjectionMatrix", "Uniform.light",
                                    "Uniform.cameraPosition" };

    const int numBlockUniforms = sizeof(uniformName) / sizeof(uniformName[0]);
#ifdef _DEBUG
    {
        GLint debugNumUniforms = 0;
        glGetProgramiv(shader, GL_ACTIVE_UNIFORMS, &debugNumUniforms);
        for (GLint i = 0; i < debugNumUniforms; ++i) {
            GLchar name[1024];
            GLsizei size = 0;
            GLenum type = GL_NONE;
            glGetActiveUniform(shader, i, sizeof(name), nullptr, &size, &type, name);
            std::cout << "Uniform #" << i << ": " << name << "\n";
        }
        assert(debugNumUniforms >= numBlockUniforms);
    }
#endif

    // Map uniform names to indices within the block
    GLuint uniformIndex[numBlockUniforms];
    glGetUniformIndices(shader, numBlockUniforms, uniformName, uniformIndex);
    assert(uniformIndex[0] < 10000);

    // Map indices to byte offsets
    GLint uniformOffset[numBlockUniforms];
    glGetActiveUniformsiv(shader, numBlockUniforms, uniformIndex, GL_UNIFORM_OFFSET, uniformOffset);
    assert(uniformOffset[0] >= 0);

    // Get texture map as generated by Vulkan
    GLuint colorTexture = GL_NONE;

    GLuint glReady, glComplete, mem;
    {
        const int textureWidth = 1024, textureHeight = 1024;
        //int textureWidth, textureHeight, channels;
        const int channels = 4;
        std::vector<std::uint8_t> data;
        //loadBMP("color.bmp", textureWidth, textureHeight, channels, data);

        glCreateTextures(GL_TEXTURE_2D, 1, &colorTexture);
        // Memory sharing crap goes here.
        // Import semaphores
        glGenSemaphoresEXT(1, &glReady);
        glGenSemaphoresEXT(1, &glComplete);

        // Platform specific import.  On non-Win32 systems use glImportSemaphoreFdEXT instead
        glImportSemaphoreWin32HandleEXT(glReady, GL_HANDLE_TYPE_OPAQUE_WIN32_EXT, vulkanExample->m_sharedResource.handles.glReady);
        glImportSemaphoreWin32HandleEXT(glComplete, GL_HANDLE_TYPE_OPAQUE_WIN32_EXT, vulkanExample->m_sharedResource.handles.glComplete);

        // Import memory
        glCreateMemoryObjectsEXT(1, &mem);
        {
            GLenum err = glGetError();
            assert(err == GL_ZERO);
        }
        // Platform specific import.  On non-Win32 systems use glImportMemoryFdEXT instead
        glImportMemoryWin32HandleEXT(mem, textureWidth * textureHeight * 4 * sizeof(byte), GL_HANDLE_TYPE_OPAQUE_WIN32_EXT,
                                     vulkanExample->m_sharedResource.handles.memory);
        {
            GLenum err = glGetError();
            assert(err == GL_ZERO);
        }
        // Use the imported memory as backing for the OpenGL texture.  The internalFormat, dimensions
        // and mip count should match the ones used by Vulkan to create the image and determine it's memory
        // allocation.
        glTextureStorageMem2DEXT(colorTexture, 1, GL_RGBA8, textureWidth, textureHeight, mem, 0);

        {
            GLenum err = glGetError();
            assert(err == GL_ZERO);
        }
    }

    GLuint trilinearSampler = GL_NONE;
    {
        glGenSamplers(1, &trilinearSampler);
        glSamplerParameteri(trilinearSampler, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glSamplerParameteri(trilinearSampler, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glSamplerParameteri(trilinearSampler, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glSamplerParameteri(trilinearSampler, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    }

#ifdef _VR
    vr::TrackedDevicePose_t trackedDevicePose[vr::k_unMaxTrackedDeviceCount];
#endif

    // Main loop:
    int timer = 0;
    while (!glfwWindowShouldClose(window)) {
        assert(glGetError() == GL_NONE);

        // Once drawing is complete, signal the Vulkan semaphore indicating
        // it can continue with it's render
        // We can continue doing this every frame; Vk does not
        // need to clear the signal or anything like that.

        GLenum dstLayout = GL_LAYOUT_SHADER_READ_ONLY_EXT;
        glSignalSemaphoreEXT(glComplete, 0, nullptr, 1, &colorTexture, &dstLayout);

        // CALL vulkan render loop
        // This will signal the glready semaphore.
        vulkanExample->doRendering(timer);

        // Semaphore should arrive from VK already signalled, so this should just work.
        // As long as we only do it once. If we do it more than once, it crashes.
        GLenum srcLayout = GL_LAYOUT_COLOR_ATTACHMENT_EXT;
        glWaitSemaphoreEXT(glReady, 0, nullptr, 1, &colorTexture, &srcLayout);

        const float nearPlaneZ = -0.1f;
        const float farPlaneZ = -100.0f;
        const float verticalFieldOfView = 45.0f * PI / 180.0f;

        Matrix4x4 eyeToHead[numEyes], projectionMatrix[numEyes], headToBodyMatrix;
#ifdef _VR
        getEyeTransformations(hmd, trackedDevicePose, nearPlaneZ, farPlaneZ, headToBodyMatrix.data, eyeToHead[0].data, eyeToHead[1].data,
                              projectionMatrix[0].data, projectionMatrix[1].data);
#else
        projectionMatrix[0] = Matrix4x4::perspective(float(framebufferWidth), float(framebufferHeight), nearPlaneZ, farPlaneZ, verticalFieldOfView);
#endif

        // printf("float nearPlaneZ = %f, farPlaneZ = %f; int width = %d, height = %d;\n", nearPlaneZ, farPlaneZ, framebufferWidth, framebufferHeight);

        const Matrix4x4& bodyToWorldMatrix =
            Matrix4x4::translate(bodyTranslation) * Matrix4x4::roll(bodyRotation.z) * Matrix4x4::yaw(bodyRotation.y) * Matrix4x4::pitch(bodyRotation.x);

        const Matrix4x4& headToWorldMatrix = bodyToWorldMatrix * headToBodyMatrix;

        for (int eye = 0; eye < numEyes; ++eye) {
            glBindFramebuffer(GL_FRAMEBUFFER, framebuffer[eye]);
            glViewport(0, 0, framebufferWidth, framebufferHeight);

            glClearColor(0.1f, 0.2f, 0.3f, 0.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            const Matrix4x4& objectToWorldMatrix =
                Matrix4x4::translate(0.0f, 0.5f, 0.0f) * Matrix4x4::yaw(PI / 3.0f) * Matrix4x4::scale(Shape::scale, Shape::scale, Shape::scale);
            const Matrix3x3& objectToWorldNormalMatrix = Matrix3x3(objectToWorldMatrix).transpose().inverse();
            const Matrix4x4& cameraToWorldMatrix = headToWorldMatrix * eyeToHead[eye];

            const Vector3& light = Vector3(1.0f, 0.5f, 0.2f).normalize();

            // Draw the background
            drawSky(framebufferWidth, framebufferHeight, nearPlaneZ, farPlaneZ, cameraToWorldMatrix.data, projectionMatrix[eye].inverse().data, &light.x);

            ////////////////////////////////////////////////////////////////////////
            // Draw a mesh
            glEnable(GL_DEPTH_TEST);
            glDepthFunc(GL_LESS);
            glEnable(GL_CULL_FACE);
            glDepthMask(GL_TRUE);

            glUseProgram(shader);

            // in position
            glBindBuffer(GL_ARRAY_BUFFER, positionBuffer);
            glVertexAttribPointer(positionAttribute, 3, GL_FLOAT, GL_FALSE, 0, 0);
            glEnableVertexAttribArray(positionAttribute);

            // in normal
            glBindBuffer(GL_ARRAY_BUFFER, normalBuffer);
            glVertexAttribPointer(normalAttribute, 3, GL_FLOAT, GL_FALSE, 0, 0);
            glEnableVertexAttribArray(normalAttribute);

            // in tangent
            if (tangentAttribute != -1) {
                // Only bind if used
                glBindBuffer(GL_ARRAY_BUFFER, tangentBuffer);
                glVertexAttribPointer(tangentAttribute, 4, GL_FLOAT, GL_FALSE, 0, 0);
                glEnableVertexAttribArray(tangentAttribute);
            }

            // in texCoord
            glBindBuffer(GL_ARRAY_BUFFER, texCoordBuffer);
            glVertexAttribPointer(texCoordAttribute, 2, GL_FLOAT, GL_FALSE, 0, 0);
            glEnableVertexAttribArray(texCoordAttribute);

            // indexBuffer
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBuffer);

            // uniform colorTexture (samplers cannot be placed in blocks)
            const GLint colorTextureUnit = 0;
            glActiveTexture(GL_TEXTURE0 + colorTextureUnit);
            glBindTexture(GL_TEXTURE_2D, colorTexture);
            glBindSampler(colorTextureUnit, trilinearSampler);
            glUniform1i(colorTextureUniform, colorTextureUnit);

            // Other uniforms in the interface block
            {
                glBindBufferBase(GL_UNIFORM_BUFFER, uniformBindingPoint, uniformBlock);

                GLubyte* ptr = (GLubyte*)glMapBuffer(GL_UNIFORM_BUFFER, GL_WRITE_ONLY);
                // mat3 is passed to openGL as if it was mat4 due to padding rules.
                for (int row = 0; row < 3; ++row) {
                    memcpy(ptr + uniformOffset[0] + sizeof(float) * 4 * row, objectToWorldNormalMatrix.data + row * 3, sizeof(float) * 3);
                }

                memcpy(ptr + uniformOffset[1], objectToWorldMatrix.data, sizeof(objectToWorldMatrix));

                const Matrix4x4& modelViewProjectionMatrix = projectionMatrix[eye] * cameraToWorldMatrix.inverse() * objectToWorldMatrix;
                memcpy(ptr + uniformOffset[2], modelViewProjectionMatrix.data, sizeof(modelViewProjectionMatrix));
                memcpy(ptr + uniformOffset[3], &light.x, sizeof(light));
                const Vector4& cameraPosition = cameraToWorldMatrix.col(3);
                memcpy(ptr + uniformOffset[4], &cameraPosition.x, sizeof(Vector3));
                glUnmapBuffer(GL_UNIFORM_BUFFER);
            }

            glDrawElements(GL_TRIANGLES, numIndices, GL_UNSIGNED_INT, 0);
#ifdef _VR
            {
                const vr::Texture_t tex = { reinterpret_cast<void*>(intptr_t(colorRenderTarget[eye])), vr::API_OpenGL, vr::ColorSpace_Gamma };
                vr::VRCompositor()->Submit(vr::EVREye(eye), &tex);
            }
#endif
        }  // for each eye

        ////////////////////////////////////////////////////////////////////////
#ifdef _VR
        // Tell the compositor to begin work immediately instead of waiting for the next WaitGetPoses() call
        vr::VRCompositor()->PostPresentHandoff();
#endif

        // Mirror to the window
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, GL_NONE);
        glViewport(0, 0, windowWidth, windowHeight);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glBlitFramebuffer(0, 0, framebufferWidth, framebufferHeight, 0, 0, windowWidth, windowHeight, GL_COLOR_BUFFER_BIT, GL_LINEAR);
        glBindFramebuffer(GL_READ_FRAMEBUFFER, GL_NONE);

        // Display what has been drawn on the main window
        glfwSwapBuffers(window);

        // Check for events
        glfwPollEvents();

        // Handle events
        if (GLFW_PRESS == glfwGetKey(window, GLFW_KEY_ESCAPE)) {
            glfwSetWindowShouldClose(window, 1);
        }

        // WASD keyboard movement
        const float cameraMoveSpeed = 0.01f;
        if (GLFW_PRESS == glfwGetKey(window, GLFW_KEY_W)) {
            bodyTranslation += Vector3(headToWorldMatrix * Vector4(0, 0, -cameraMoveSpeed, 0));
        }
        if (GLFW_PRESS == glfwGetKey(window, GLFW_KEY_S)) {
            bodyTranslation += Vector3(headToWorldMatrix * Vector4(0, 0, +cameraMoveSpeed, 0));
        }
        if (GLFW_PRESS == glfwGetKey(window, GLFW_KEY_A)) {
            bodyTranslation += Vector3(headToWorldMatrix * Vector4(-cameraMoveSpeed, 0, 0, 0));
        }
        if (GLFW_PRESS == glfwGetKey(window, GLFW_KEY_D)) {
            bodyTranslation += Vector3(headToWorldMatrix * Vector4(+cameraMoveSpeed, 0, 0, 0));
        }
        if (GLFW_PRESS == glfwGetKey(window, GLFW_KEY_C)) {
            bodyTranslation.y -= cameraMoveSpeed;
        }
        if ((GLFW_PRESS == glfwGetKey(window, GLFW_KEY_SPACE)) || (GLFW_PRESS == glfwGetKey(window, GLFW_KEY_Z))) {
            bodyTranslation.y += cameraMoveSpeed;
        }

        // Keep the camera above the ground
        if (bodyTranslation.y < 0.01f) {
            bodyTranslation.y = 0.01f;
        }

        static bool inDrag = false;
        const float cameraTurnSpeed = 0.005f;
        if (GLFW_PRESS == glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT)) {
            static double startX, startY;
            double currentX, currentY;

            glfwGetCursorPos(window, &currentX, &currentY);
            if (inDrag) {
                bodyRotation.y -= float(currentX - startX) * cameraTurnSpeed;
                bodyRotation.x -= float(currentY - startY) * cameraTurnSpeed;
            }
            inDrag = true;
            startX = currentX;
            startY = currentY;
        } else {
            inDrag = false;
        }

        ++timer;
    }

#ifdef _VR
    if (hmd != nullptr) {
        vr::VR_Shutdown();
    }
#endif

    // Close the GL context and release all resources
    glfwTerminate();
    delete (vulkanExample);

    return 0;
}
#endif
