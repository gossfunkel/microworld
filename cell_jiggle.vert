#version 430
#extension GL_GOOGLE_include_directive : require

// SSBO for vertex pulling
#include  "cell.glsl"
layout (std430, binding = 0) buffer ssbo { 
    P3d_data p3d_data[13];          //  = 64B x 13 
};                                  // = 832B buffer

uniform mat4 p3d_ModelViewProjectionMatrix;
uniform float osg_DeltaFrameTime;

const float EPSILON = 0.0001;               // a really small number
const float DAMP_RATIO = .3;                // sets springyness of cell verts

uniform float radius;
uniform vec2 model_velocity;
//uniform vec4 col;

// disabled while testing tess shaders
//out vec4 v_col;

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
    vec4 new_pos = vec4(0.,0.,0.,1.);       // default value for centrepoint, 
    if (vtx != 0) {                         // which experiences no harmonic motion
        // get the 2D vertex positions in world-space
        //vec2 centrepoint = vec4(p3d_ModelMatrix * vec4(p3d_data[0].pos,0.,1.)).xy;
        //vec2 vtx_world = vec4(p3d_ModelMatrix * p3d_data[vtx].pos).xy;

        // get change in origin position 
        //vec2 model_speed = vtx_data[0].vel;

        // calculate vertex position relative to centrepoint after movement
        vec2 vtx_mod = p3d_data[vtx].pos.xy - model_velocity;

        // calculate where potential minimum is for vertex based on relative displacement of centrepoint over dt
        // desire position = relative-centrepoint-pos + basis-vector-to-centrepoint * radius
        vec2 desire_vtx = p3d_data[vtx].basis * radius;
        
        // calculate new position and vel in 2D model-space
        vec4 sho_out = SHO(vtx_mod, 
                      p3d_data[vtx].vel, 
                      desire_vtx, 
                      osg_DeltaFrameTime,
                      10.);
        new_pos = vec4(sho_out.xy, 0., 1.);
        // calculate model translation in world-space
        //vec2 model_pos = vec4(p3d_ModelMatrix * vec4(vtx_data[vtx].basis,0.,1.)).xy;
        // transform to world-space
        p3d_data[vtx].pos = new_pos;

        // write output to buffers (FIXME relativity??)
        p3d_data[vtx].vel = sho_out.zw;
        // new_pos = vec4(desire_vtx,0.,1.);
        //p3d_data[vtx].pos = inverse(p3d_ModelMatrix) * new_pos;
    } 
    
    // calculate gl_Position with the new position and remaining matrices
    gl_Position = p3d_ModelViewProjectionMatrix * new_pos;
}