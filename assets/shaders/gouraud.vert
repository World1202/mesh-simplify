#version 330 core

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;
uniform mat3 u_normal_matrix;

uniform vec3 u_camera_position;
uniform vec3 u_light_position;
uniform vec3 u_light_color;
uniform vec3 u_fill_light_position;
uniform vec3 u_fill_light_color;
uniform float u_fill_light_strength;
uniform vec3 u_object_color;
uniform float u_ambient_strength;
uniform float u_diffuse_strength;
uniform float u_specular_strength;
uniform float u_shininess;

out vec3 v_color;

void main() {
    vec4 world_position = u_model * vec4(a_position, 1.0);
    vec3 world_normal   = normalize(u_normal_matrix * a_normal);
    vec3 view_direction = normalize(u_camera_position - world_position.xyz);

    // ── 主光源（漫反射 + 镜面反射） ──
    vec3  main_dir     = normalize(u_light_position - world_position.xyz);
    vec3  reflect_dir  = reflect(-main_dir, world_normal);
    float diffuse_fac  = max(dot(world_normal, main_dir), 0.0);
    float specular_fac = pow(max(dot(view_direction, reflect_dir), 0.0), u_shininess);

    vec3 ambient  = u_ambient_strength  * u_light_color * u_object_color;
    vec3 diffuse  = u_diffuse_strength  * diffuse_fac  * u_light_color * u_object_color;
    vec3 specular = u_specular_strength * specular_fac * u_light_color;

    // ── 补光（仅漫反射，无镜面，避免双高光） ──
    vec3  fill_dir = normalize(u_fill_light_position - world_position.xyz);
    float fill_fac = max(dot(world_normal, fill_dir), 0.0);
    vec3  fill     = u_fill_light_strength * fill_fac * u_fill_light_color * u_object_color;

    v_color = ambient + diffuse + specular + fill;

    gl_Position = u_projection * u_view * world_position;
}
