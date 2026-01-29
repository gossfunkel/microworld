#version 430

in vec4 v_col;

// out to screen
out vec4 p3d_FragColor;

void main() {
    p3d_FragColor = uvec4(255 * v_col.x,
                          255 * v_col.y,
                          255 * v_col.z,
                          255 * v_col.w);
}
