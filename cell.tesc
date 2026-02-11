#version 430

//          (TCS) TESSELLATION CONTROL: define number of verts to make with tessellation

//      Tessellation-Control-Shader-specific variables:
//gl_PatchVerticesIn // verts per patch
//gl_PrimitiveID     // current patch
//gl_InvocationID    // current vertex within patch


//      gets these inputs from vertex shader:
//in gl_PerVertex {
//  vec4 gl_Position;
//  float gl_PointSize;
//  float gl_ClipDistance[];
//} gl_in[gl_MaxPatchVertices];

// per vertex outputs go in arrays e.g.
//out vec2 vertexTexCoord[];
// patch output is specified individually
//patch out vec4 data;

//      built-in outputs: 
//patch out float gl_TessLevelOuter[4];
//patch out float gl_TessLevelInner[2];
//out gl_PerVertex {
//  vec4 gl_Position;
//  float gl_PointSize;
//  float gl_ClipDistance[];
//} gl_out[];

uniform float lod_level;

//layout (location = 0) in Vtx_data patchout[];
layout(vertices = 3) out;
//layout(location = 1) out Vtx_data vtx_tesc[];

//in vec2 texcoords_pch[];
//out vec2 texcoords_vtx[];

// FIXME needs to be array
//in vec4 v_col;
//out vec4 vtx_col;

void main() {
    //gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;
    //texcoords_pch[gl_InvocationID] = texcoords_vtx[gl_InvocationID];

    // call barrier() to sync between invocations

    // tessellate outer edge only
    gl_TessLevelOuter[0] = 3;
    gl_TessLevelOuter[1] = lod_level;

    // only tesselate once per triangle
    if (gl_InvocationID == 0) {
        gl_out[gl_InvocationID].gl_Position = gl_in[0].gl_Position;
    } else if (gl_InvocationID == 1) {
        gl_out[gl_InvocationID].gl_Position = gl_in[gl_PrimitiveID+1].gl_Position;
    } else if (gl_InvocationID == 2) {
        gl_out[gl_InvocationID].gl_Position = gl_in[gl_PrimitiveID+2].gl_Position;
    }

    // pass vertex positions
    //vtx_col = v_col;
}