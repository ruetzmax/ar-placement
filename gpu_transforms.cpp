#include <iostream>
#include <string>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

std::string defaultVertexShader = R"(
# version 330 core
layout ( location = 0) in vec2 aPos ;
layout ( location = 1) in vec2 aTexCoord ;
out vec2 TexCoord ;
void main () {
gl_Position = vec4 ( aPos , 0.0 , 1.0) ;
TexCoord = aTexCoord ;
}
)";

std::string defaultFragmentShader = R"(
# version 330 core
out vec4 FragColor ;
in vec2 TexCoord ;
uniform sampler2D texture1 ;
void main () {
FragColor = texture ( texture1 , vec2(TexCoord.x, 1.0 - TexCoord.y) ) ;
}
)";

std::string pixelateFragmentShader = R"(
# version 330 core
out vec4 FragColor ;
in vec2 TexCoord ;
uniform sampler2D texture1 ;
void main () {
FragColor = vec4(1.0, 0.0, 0.0, 1.0);
}
)";

unsigned int compileShader(unsigned int type, const std::string& source)
{
    unsigned int id = glCreateShader(type);
    const char* src = source.c_str();
    glShaderSource(id, 1, &src, nullptr);
    glCompileShader(id);

    // Error handling
    int result;
    glGetShaderiv(id, GL_COMPILE_STATUS, &result);
    if (result == GL_FALSE)
    {
        int length;
        glGetShaderiv(id, GL_INFO_LOG_LENGTH, &length);
        char* message = (char*)alloca(length * sizeof(char));
        glGetShaderInfoLog(id, length, &length, message);
        std::cout << "Failed to compile " << (type == GL_VERTEX_SHADER ? "vertex" : "fragment") << " shader!" << std::endl;
        std::cout << message << std::endl;
        glDeleteShader(id);
        return 0;
    }

    return id;
}

unsigned int createShaderProgram(const std::string& vertexShader, const std::string& fragmentShader)
{
    unsigned int program = glCreateProgram();
    unsigned int vs = compileShader(GL_VERTEX_SHADER, vertexShader);
    unsigned int fs = compileShader(GL_FRAGMENT_SHADER, fragmentShader);

    glAttachShader(program, vs);
    glAttachShader(program, fs);
    glLinkProgram(program);
    glValidateProgram(program);

    glDeleteShader(vs);
    glDeleteShader(fs);

    return program;
}

unsigned int defaultShaderProgram = 0;
unsigned int pixelateShaderProgram = 0;

void initShaderPrograms()
{
    if (defaultShaderProgram != 0 || pixelateShaderProgram != 0)
        return; // already initialized

    defaultShaderProgram = createShaderProgram(defaultVertexShader, defaultFragmentShader);
    pixelateShaderProgram = createShaderProgram(defaultVertexShader, pixelateFragmentShader);
}

void cleanupShaderPrograms()
{
    if (defaultShaderProgram != 0)
    {
        glDeleteProgram(defaultShaderProgram);
        defaultShaderProgram = 0;
    }
    if (pixelateShaderProgram != 0)
    {
        glDeleteProgram(pixelateShaderProgram);
        pixelateShaderProgram = 0;
    }
}
    