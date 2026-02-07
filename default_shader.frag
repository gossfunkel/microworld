#version 430

in vec4 v_col;

// out to screen
out vec4 p3d_FragColor;

void main() {
    p3d_FragColor = v_col;
}
