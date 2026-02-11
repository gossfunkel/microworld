#version 430

//          (TES) TESSELATION EVALUATION: define position of new tessellated verts

//      Tessellation-Eval-Shader-specific variables:
//gl_PatchVerticesIn // verts per patch
//gl_PrimitiveID     // current patch
//gl_TessCoord       // current vertex within patch

//      gets these inputs from TCS:
//patch in float gl_TessLevelOuter[4];
//patch in float gl_TessLevelInner[2];
//in gl_PerVertex {
//  vec4 gl_Position;
//  float gl_PointSize;
//  float gl_ClipDistance[];
//} gl_in[gl_MaxPatchVertices];

//      built-in outputs: 
//out gl_PerVertex {
//  vec4 gl_Position;
//  float gl_PointSize;
//  float gl_ClipDistance[];
//}

uniform mat4 p3d_ModelViewProjectionMatrix;

// input prim type, spacing (equal_spacing, 
//      fractional_even_spacing, or fractional_odd_spacing), 
//      prim gen/winding order (cw or ccw)
layout (triangles, equal_spacing, ccw) in;

//layout (vertices = 2) in;
//layout (vertices = 2) out;

/* by fcaruso, computeranimations.wordpress.com
vec3 hermite(float u, vec3 p0, vec3 p1, vec3 t0, vec3 t1) {
    float F1 = 2.*u*u*u - 3.*u*u + 1.;
    float F2 = -2.*u*u*u + 3*u*u;
    float F3 = u*u*u - 2.*u*u + u;
    float F4 = u*u*u - u*u;

    return F1*p0 + F2*p1 + F3*t0 + F4*t1;
}*/

/* by ExtraGravity, enochtsang.com
vec4 catmull_rom(float u, float v, float p0, float p1, float p_1, float p2) {
    float b0 = (2. * u * u) - (u * u * u) - u;
    float b1 = 2. - (5. * u * u) + (3. * u * u * u);
    float b2 = (u) + (4. * u * u) - (3. * u * u * u);
    float b3 = (u * u * u) - (u * u);
    vec4 new_pos = .5 * (b0*p_1 + b1*p0 + b2*p1 + b3*p2);
    return vec4(new_pos.x + v * .08, new_pos.y + v * .08, new_pos.z, new_pos.w);
}*/

// FIXME needs to be an array
//in vec4 vtx_col;
//out vec4 v_col;

void main() {
    vec4 centre = gl_in[0].gl_Position;
    vec4 p1 = gl_in[1].gl_Position;
    vec4 p2 = gl_in[2].gl_Position;
    float u = gl_TessCoord.x;
    float v = gl_TessCoord.y;

    // do hermitic splinic interpolation
/*    vec3 vPos0 = vec3( gl_in[0].gl_Position );
    vec3 vPos1 = vec3( gl_in[1].gl_Position );
    vec3 v3pos = hermite( u, vPos0, vPos1, vTan0, vTan1 );
    vec4 pos = vec4( v3pos, 1.); */
    // do catmull-rom interpolation
    //vec4 pos = catmull_rom(u,v,p0,p1)

    // do radial value interpolation
    float p1_dR = gl_in[1].gl_PointSize * ((gl_PatchVerticesIn-u)/gl_PatchVerticesIn);
    float p2_dR = gl_in[2].gl_PointSize * (u/gl_PatchVerticesIn);
    float dR = p1_dR + p2_dR;
    vec4 norm = p1 * ((gl_PatchVerticesIn-u)/gl_PatchVerticesIn) + p2 * (u/gl_PatchVerticesIn);
    vec4 pos = norm * dR;

    // apply matrices 
    gl_Position = p3d_ModelViewProjectionMatrix * pos;
    //v_col = vtx_col;
}