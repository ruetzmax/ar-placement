#pragma once

#include <string>
#include <GL/glew.h>

unsigned int compileShader(unsigned int type, const std::string& source);
unsigned int createShaderProgram(const std::string& vertexShader, const std::string& fragmentShader);
unsigned int getShaderProgram(int filter, bool isInteractive);
void initShaderPrograms();
void cleanupShaderPrograms();
