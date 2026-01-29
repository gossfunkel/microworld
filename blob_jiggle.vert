#version 430

//uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ViewProjectionMatrix;
uniform mat4 p3d_ModelMatrix;
//uniform float4x4 trans_world_to_blob;
uniform float osg_DeltaFrameTime;

// nb 'inout' is not in-standard, and it only passes the value to the frag shader
/*inout vec4 p3d_Vertex;
in vec4 p3d_Color;
in vec2 basis;*/

// SSBO for vertex pulling
layout (std430, binding = 0) buffer ssbo { 
    // 11 float32s and 4 int8s - 384b; 12 * scalar (4 bits)
    vec4 p3d_Vertex[13];
    vec3 p3d_Normal[13];
    float size[13];
    uvec4 p3d_Color[13];
    vec2 basis[13];
    vec2 velocity[13]; 
};

const float EPSILON = 0.0001;
const float DAMP_RATIO = .6;              // sets springyness of object
//const float DIST_EDGEPOINTS = .51;        // hopefully should be compatible with the radius
uniform float radius;

out vec4 col;

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
    col = p3d_Color[vtx];
    vec4 new_pos;
    if (vtx != 0) {
        // get vertex position in world-space
        vec2 centrepoint = vec4(p3d_ModelMatrix * vec4(0.,0.,0.,1.)).xy;
        vec2 vtx_world = vec4(p3d_ModelMatrix * p3d_Vertex[vtx]).xy;

        // calculate equilibrium pos in world-space
        vec2 desire_vtx = centrepoint + basis[vtx].xy * radius;

        // calculate new position and vel in 2D world-space
        vec4 sho_out = SHO(vtx_world, 
                      velocity[vtx], 
                      desire_vtx, 
                      osg_DeltaFrameTime,
                      10.);
        new_pos = vec4(sho_out.xy, 0., 1.);// *  inverse(p3d_ModelMatrix);

        // write output to buffers (FIXME relativity??)
        velocity[vtx] = sho_out.zw;
    } else {
        // centrepoint experiences no physics - just move basis to world-space
        new_pos = p3d_ModelMatrix * vec4(basis[vtx],0.,1.);
    }
    
    p3d_Vertex[vtx] = inverse(p3d_ModelMatrix) * new_pos;
    // calculate gl_Position with the new position and remaining matrices
    gl_Position = p3d_ViewProjectionMatrix * new_pos;
}