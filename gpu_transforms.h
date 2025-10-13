#pragma once

#include <string>
#include <GL/glew.h>

extern unsigned int defaultShaderProgram;
extern unsigned int pencilShaderProgram;

unsigned int compileShader(unsigned int type, const std::string& source);
unsigned int createShaderProgram(const std::string& vertexShader, const std::string& fragmentShader);
void initShaderPrograms();
void cleanupShaderPrograms();
