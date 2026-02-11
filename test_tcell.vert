#version 430
#extension GL_GOOGLE_include_directive : require

uniform float osg_DeltaFrameTime;

// SSBO for vertex pulling
#include "cell.glsl"
layout (std430, binding = 0) buffer ssbo { 
    P3d_data p3d_data[13];          //  = 64B x 13 
};                                  // = 832B buffer

const float EPSILON = 0.0001;
const float DAMP_RATIO = .3;              // sets springyness of object
uniform float radius;
uniform vec2 model_velocity;
//uniform int num_vtxs;
//uniform vec4 col;

//out vec4 v_col;
out float dPos[13];

//layout (location = 0) out Vtx_data patchout;

vec4 SHO(vec2 pos, vec2 vel, vec2 equilibriumPos, float deltaTime, float angularFreq) {
    // SHM angular frequency parameter must be positive!
    // SHM damping ratio parameter must be positive!
    float pospos;
    float posvel;
    float velpos;
    float velvel;

    if (angularFreq < EPSILON) {
        // SHM frequency too low to change motion
        pospos = 1.;
        posvel = 0.;
        velpos = 0.;
        velvel = 1.;
    } else {
        if (DAMP_RATIO > 1. + EPSILON) {
            // overdamped formula
            float za = -angularFreq * DAMP_RATIO;
            float zb = angularFreq * sqrt(abs(DAMP_RATIO*DAMP_RATIO - 1.));
            float z1 = za - zb;
            float z2 = za + zb;

            float e1 = exp(z1 * deltaTime);
            float e2 = exp(z2 * deltaTime);

            float invTwoZb = 1. / (2. * zb);

            float e1OverTwoZb = e1 * invTwoZb;
            float e2OverTwoZb = e2 * invTwoZb;

            float z1e1OverTwoZb = z1 * e1OverTwoZb;
            float z2e2OverTwoZb = z2 * e2OverTwoZb;

            pospos = e1OverTwoZb * z2e2OverTwoZb + e2OverTwoZb;
            posvel = -e1OverTwoZb + e2OverTwoZb;
            velpos = (z1e1OverTwoZb - z2e2OverTwoZb + e2) * z2;
            velvel = -z1e1OverTwoZb + z2e2OverTwoZb;
        } else if (DAMP_RATIO < 1. - EPSILON) {
            // underdamped formula
            float omegaZeta = angularFreq * DAMP_RATIO;
            float alpha     = angularFreq * sqrt(1. - DAMP_RATIO * DAMP_RATIO);

            float expTerm = exp(-omegaZeta * deltaTime);
            float cosTerm = cos(alpha * deltaTime);
            float sinTerm = sin(alpha * deltaTime);

            float invAlpha = 1. / alpha;

            float expSin = expTerm * sinTerm;
            float expCos = expTerm * cosTerm;
            float expOmegaZetaSinOverAlpha = expTerm * omegaZeta * sinTerm * invAlpha;

            pospos = expCos + expOmegaZetaSinOverAlpha;
            posvel = expSin * invAlpha;
            velpos = -expSin * alpha - omegaZeta * expOmegaZetaSinOverAlpha;
            velvel = expCos - expOmegaZetaSinOverAlpha;
        } else {
            // critically damped formula
            float expTerm = exp(-angularFreq * deltaTime);
            float timeExp = deltaTime * expTerm;
            float timeExpFreq = timeExp * angularFreq;

            pospos = timeExpFreq + expTerm;
            posvel = timeExp;
            velpos = -angularFreq * timeExpFreq;
            velvel = -timeExpFreq + expTerm;
        }
    }

    pos = pos - equilibriumPos;
    vec2 oldvel = vel;
    vel = pos * velpos + oldvel * velvel;
    pos = pos * pospos + oldvel * posvel + equilibriumPos;

    return vec4(pos, vel);
}

void main() {
    uint vtx = gl_VertexID;
    //v_col = p3d_data[vtx].colour * col;
    // calculate vertex position relative to model pos after movement
    vec2 vtx_mod = p3d_data[vtx].pos.xy - model_velocity;
    vec2 desire_vtx = p3d_data[vtx].basis * radius;
    
    // calculate new position and vel in 2D model-space
    vec4 sho_out = SHO(vtx_mod, 
                  p3d_data[vtx].vel, 
                  desire_vtx, 
                  osg_DeltaFrameTime,
                  10.);
    vec4 new_pos = vec4(sho_out.xy, 0., 1.);

    // calculate drad (in 2D) for tessellation interpolation
    vec2 diff_pos = vec2(new_pos.xy - vtx_mod.xy);
    dPos[vtx] = diff_pos.x*diff_pos.x+diff_pos.y*diff_pos.y;

    // write output to buffers (FIXME relativity??)
    p3d_data[vtx].pos = new_pos;
    p3d_data[vtx].vel = sho_out.zw;
    
    // calculate gl_Position with the new position and apply matrices
    //patchout.position = new_pos;
    // pass data to tess shader
    //patchout.normal   = p3d_data[vtx].normal;
    //patchout.colour   = p3d_data[vtx].colour;
    // update gl_pos
    //gl_Position = patchout.position;
    gl_Position = new_pos;
}