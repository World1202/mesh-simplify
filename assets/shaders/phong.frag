#version 330 core

in vec3 v_world_position;
in vec3 v_world_normal;

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

out vec4 fragColor;

void main() {
    vec3 normal          = normalize(v_world_normal);
    vec3 view_direction  = normalize(u_camera_position - v_world_position);

    // ── 主光源（漫反射 + 镜面反射） ──
    vec3  main_dir      = normalize(u_light_position - v_world_position);
    vec3  reflect_dir   = reflect(-main_dir, normal);
    float diffuse_fac   = max(dot(normal, main_dir), 0.0);
    float specular_fac  = pow(max(dot(view_direction, reflect_dir), 0.0), u_shininess);

    vec3 ambient  = u_ambient_strength  * u_light_color * u_object_color;
    vec3 diffuse  = u_diffuse_strength  * diffuse_fac  * u_light_color * u_object_color;
    vec3 specular = u_specular_strength * specular_fac * u_light_color;

    // ── 补光（仅漫反射，无镜面，避免双高光） ──
    vec3  fill_dir    = normalize(u_fill_light_position - v_world_position);
    float fill_fac    = max(dot(normal, fill_dir), 0.0);
    vec3  fill        = u_fill_light_strength * fill_fac * u_fill_light_color * u_object_color;

    fragColor = vec4(ambient + diffuse + specular + fill, 1.0);
}
