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
FragColor = texture ( texture1 , vec2(TexCoord.x, TexCoord.y) ) ;
}
)";

std::string pencilFragmentShader = R"(
# version 330 core
out vec4 FragColor ;
in vec2 TexCoord ;
uniform sampler2D texture1 ;

uniform int kernelRadius;
uniform float weights[1000];

void main () {
    ivec2 texSize = textureSize(texture1, 0);
    vec2 tex_offset = 1.0 / vec2(texSize);

    // determine pixel gray scale value
    float grayScale = dot(texture(texture1, TexCoord).rgb, vec3(0.299, 0.587, 0.114));

    // apply gaussian blur
    float blurredGrayScale = 0.0;
    float weightSum = 0.0;

    for(int y = -kernelRadius / 2; y <= kernelRadius / 2; y++)
    {
        for(int x = -kernelRadius / 2; x <= kernelRadius / 2; x++)
        {
            vec2 offset = vec2(float(x), float(y)) * tex_offset;
            vec3 color = texture(texture1, TexCoord + offset).rgb;
            float sampleGray = dot(color, vec3(0.299, 0.587, 0.114));
            sampleGray = 1.0 - sampleGray;
            float weight = weights[(y + kernelRadius / 2) * kernelRadius + (x + kernelRadius / 2)];
            blurredGrayScale += sampleGray * weight;
            weightSum += weight;
        }
    }

    blurredGrayScale = 1.0 - blurredGrayScale;
    
    blurredGrayScale /= weightSum; // normalize
    float outputColor = clamp(grayScale / (blurredGrayScale + 0.0001), 0.0, 1.0);

    FragColor = vec4(outputColor, outputColor, outputColor, 1.0);
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
unsigned int pencilShaderProgram = 0;

void initShaderPrograms()
{
    if (defaultShaderProgram != 0 || pencilShaderProgram != 0)
        return;

    defaultShaderProgram = createShaderProgram(defaultVertexShader, defaultFragmentShader);
    pencilShaderProgram = createShaderProgram(defaultVertexShader, pencilFragmentShader);
}

void cleanupShaderPrograms()
{
    if (defaultShaderProgram != 0)
    {
        glDeleteProgram(defaultShaderProgram);
        defaultShaderProgram = 0;
    }
    if (pencilShaderProgram != 0)
    {
        glDeleteProgram(pencilShaderProgram);
        pencilShaderProgram = 0;
    }
}
    