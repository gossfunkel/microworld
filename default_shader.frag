#version 430
in vec4 p3d_Color;

// out to screen
out vec4 p3d_FragColor;

void main() {
    p3d_FragColor = p3d_Color;
}
