#version 430

in vec4 col;

// out to screen
out vec4 p3d_FragColor;

void main() {
    p3d_FragColor = uvec4(255*col.x,255*col.y,255*col.z,255*col.w);
}
