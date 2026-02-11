#ifndef _CELL_H
#define _CELL_H

struct P3d_data {
    vec4 pos;                       // 4x 4B
    vec4 normal;                    // 4x 4B
    vec4 colour;                    // 4x 4B
    vec2 basis;                     // 2x 4B
    vec2 vel;                       // 2x 4B
};                                  //  = 16B

struct Vtx_data {
    vec4 pos;                       // 4x 4B
    vec4 normal;                    // 4x 4B
    vec4 colour;                    // 4x 4B
};

#undef _CELL_H
#endif