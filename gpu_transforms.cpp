#include <iostream>
#include <string>
#include <map>
#include <utility>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

std::string defaultVertexShader = R"(
#version 330 core
layout ( location = 0) in vec2 aPos ;
layout ( location = 1) in vec2 aTexCoord ;
out vec2 TexCoord ;
void main () {
gl_Position = vec4 ( aPos , 0.0 , 1.0) ;
TexCoord = aTexCoord ;
}
)";

std::string transformVertexShader = R"(
#version 330 core
layout ( location = 0) in vec2 aPos ;
layout ( location = 1) in vec2 aTexCoord ;
out vec2 TexCoord ;
uniform float rotX;
uniform float rotY;
uniform float posX;
uniform float posY;
uniform float scale;
void main () {
    float focalLength = 500.0;
    float cosX = cos(rotX * 3.14159265 / 180.0);
    float sinX = sin(rotX * 3.14159265 / 180.0);
    float cosY = cos(rotY * 3.14159265 / 180.0);
    float sinY = sin(rotY * 3.14159265 / 180.0);

    mat3 rotXMat = mat3(
        1, 0, 0,
        0, cosX, -sinX,
        0, sinX, cosX);

    mat3 rotYMat = mat3(
        cosY, 0, sinY,
        0, 1, 0,
        -sinY, 0, cosY);

    mat3 transMat = mat3(
        1, 0, posX,
        0, 1, posY,
        0, 0, 1);

    mat3 scaleMat = mat3(
        scale, 0, 0,
        0, scale, 0,
        0, 0, 1);

    vec3 newPos = scaleMat * rotYMat * rotXMat * vec3(aPos, 0.0);
    newPos += vec3(posX, posY, focalLength);
    newPos = vec3(newPos.x * focalLength / newPos.z, newPos.y * focalLength / newPos.z, 1.0);

    gl_Position = vec4(newPos.x, newPos.y, 1.0, 1.0);

    TexCoord = aTexCoord;
}
)";

std::string defaultFragmentShader = R"(
#version 330 core
out vec4 FragColor ;
in vec2 TexCoord ;
uniform sampler2D texture1 ;
void main () {
FragColor = texture ( texture1 , vec2(TexCoord.x, TexCoord.y) ) ;
}
)";

std::string pencilFragmentShader = R"(
#version 330 core
out vec4 FragColor ;
in vec2 TexCoord ;
uniform sampler2D texture1 ;

uniform int kernelSize;
uniform float weights[1000];

void main () {
    ivec2 texSize = textureSize(texture1, 0);
    vec2 tex_offset = 1.0 / vec2(texSize);

    // determine pixel gray scale value
    float grayScale = dot(texture(texture1, TexCoord).rgb, vec3(0.299, 0.587, 0.114));

    // apply gaussian blur
    float blurredGrayScale = 0.0;
    float weightSum = 0.0;

    for(int y = -kernelSize / 2; y <= kernelSize / 2; y++)
    {
        for(int x = -kernelSize / 2; x <= kernelSize / 2; x++)
        {
            vec2 offset = vec2(float(x), float(y)) * tex_offset;
            vec3 color = texture(texture1, TexCoord + offset).rgb;
            float sampleGray = dot(color, vec3(0.299, 0.587, 0.114));
            sampleGray = 1.0 - sampleGray;
            float weight = weights[(y + kernelSize / 2) * kernelSize + (x + kernelSize / 2)];
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

std::string retroFragmentShader = R"(
    #version 330 core
    out vec4 FragColor ;
    in vec2 TexCoord ;
    uniform sampler2D texture1 ;
    uniform int blockPixelSize ;
    uniform int colorDepth ;

    void main () {
        ivec2 texSize = textureSize(texture1, 0);

        vec2 tex_offset = 1.0 / vec2(texSize);
        vec2 blockSize = vec2(float(blockPixelSize) / float(texSize.x), float(blockPixelSize) / float(texSize.y));

        // sample top-left pixel of block
        ivec2 currentBlockIndex = ivec2(floor(TexCoord.x / blockSize.x), floor(TexCoord.y / blockSize.y));
        vec2 currentBlockStart = currentBlockIndex * blockSize;

        vec3 avgCol = texture(texture1, currentBlockStart).rgb;

        // reduce color depth
        int levels = 1 << colorDepth;
        float step = 1.0 / float(levels);
        avgCol = floor(avgCol / step) * step;

        FragColor = vec4(avgCol, 1.0);
    }

)";

// Map to store shader programs for all combinations of vertex and fragment shaders
// Key: pair<vertex_type, fragment_type> where types are: 0=default, 1=transform for vertex; 0=default, 1=pencil, 2=retro for fragment
std::map<std::pair<int, int>, unsigned int> shaderProgramMap;

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

unsigned int getShaderProgram(int filter, bool isInteractive){
    int vertexType = isInteractive ? 1 : 0;
    int fragmentType = filter;
    
    auto key = std::make_pair(vertexType, fragmentType);
    auto it = shaderProgramMap.find(key);
    
    if (it != shaderProgramMap.end()) {
        return it->second;
    }
    
    std::cout << "Error: Shader program not found for vertex type " << vertexType 
              << " and fragment type " << fragmentType << std::endl;
    return 0;
}

void initShaderPrograms()
{    
    // Create all combinations of vertex and fragment shaders
    std::string* vertexShaders[] = { &defaultVertexShader, &transformVertexShader };
    std::string* fragmentShaders[] = { &defaultFragmentShader, &pencilFragmentShader, &retroFragmentShader };
    
    for (int v = 0; v < 2; v++) {
        for (int f = 0; f < 3; f++) {
            auto key = std::make_pair(v, f);
            unsigned int program = createShaderProgram(*vertexShaders[v], *fragmentShaders[f]);
            shaderProgramMap[key] = program;
        }
    }
}

void cleanupShaderPrograms()
{
    for (auto& pair : shaderProgramMap) {
        if (pair.second != 0) {
            glDeleteProgram(pair.second);
        }
    }
    shaderProgramMap.clear();
}
    