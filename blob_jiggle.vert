#version 430

uniform mat4 p3d_ModelViewProjectionMatrix;

in vec4 p3d_Vertex;
// TODO get other vertices
// TODO buffer for velocities

//in vec2 texcoord;
//out vec2 vtexcoord;

void main() {
    // scale with matrices
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
    // TODO modify vertex position based on centrepoint and velocity

    // TODO textures
    //vtexcoord = texcoord;
}