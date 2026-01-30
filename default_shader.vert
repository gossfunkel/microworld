#version 430

uniform mat4 p3d_ModelViewProjectionMatrix;

in vec4 p3d_Color;

uniform vec4 col;

out vec4 v_col;

// buffers:
struct P3d_data {
    vec4 pos;                       // 4x 4B
    vec4 normal;                    // 4x 4B
    vec4 colour;                    // 4x 4B
    vec2 basis;                     // 2x 4B
    vec2 vel;                       // 2x 4B
};                              //  = 16x 4B = 64B

// SSBO containing vertex data as above struct
layout (std430, binding = 0) buffer ssbo { 
    P3d_data p3d_data[13];          // 13x 64B
};                                  //  = 832B buffer

void main() {
    uint vtx = gl_VertexID;
    v_col = p3d_data[vtx].colour * col;

    gl_Position = p3d_ModelViewProjectionMatrix * p3d_data[vtx].pos;
}