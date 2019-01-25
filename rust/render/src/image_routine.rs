use rustonum::MatrixXf;
use rustonum::{
    Vector2f, 
    Vector3f,
    GeometryVector,
    Vector
};

use geometry_routine::{
    is_point_inside_triangle, 
    compute_barycentric_coords2d
};


use itertools::Itertools;

pub fn restrict_vector2f(canvas: &MatrixXf, vec: &Vector2f) -> (usize, usize)
{
    let x = if vec.x() < 0.0
            {0} 
        else if vec.x() >= (canvas.cols() as f32)
            {canvas.cols() - 1}
        else 
            {vec.x() as usize};
    let y = if vec.y() < 0.0
            {0} 
        else if vec.y() >= (canvas.rows() as f32)
            {canvas.rows() - 1}
        else 
            {vec.y() as usize};
    (x, y)
}

pub fn is_vector2f_inside_canvas(canvas: &MatrixXf, vec: Vector2f) -> bool
{
    let row = vec.y();
    let col = vec.x();
    vec.y() >= 0.0 && canvas.rows() as f32 > row &&
    vec.x() >= 0.0 && canvas.cols() as f32 > col 
}

pub fn draw_pointf(canvas: & mut MatrixXf, vec: Vector2f, val: f32)
{
    let urow = vec.y() as usize;
    let ucol = vec.x() as usize;
    if is_vector2f_inside_canvas(canvas, vec){
        canvas[(urow, ucol)] = val;
    }
}

pub fn draw_line(canvas : & mut MatrixXf, p1 : &Vector2f, p2 : &Vector2f)
{
    let dx  = p2.x() - p1.x();
    let dy  = p2.y() - p1.y();
    let a   = dy/dx;
    let b   = p1.y() - a * p1.x();
    let dxn = if dx > 0.0 {1.0} else {-1.0};
    let dyn = if dy > 0.0 {1.0} else {-1.0};
    let mut current = p1.clone();
    while (current.x() - p2.x()).abs() > 0.1 && (current.y() -p2.y()).abs() > 0.1 {
        let err1 = (dy * (current.x() + dxn) - dx * (current.y()) + dx * b).abs();
        let err2 = (dy * (current.x()) - dx * (current.y() + dyn) + dx * b).abs();
        let err3 = (dy * (current.x() + dxn) - dx * (current.y() + dyn) + dx * b).abs();
        if err1 <= err2 && err1 <= err3 {
            current[0] += dxn;
        }else if err2 <= err1 && err2 <= err3 {
            current[1] += dyn;
        }else if err3 <= err2 && err3 <= err1 {
            current[0] += dxn;
            current[1] += dyn;
        }
        draw_pointf(canvas, current, 255.0);
    }
}

pub fn draw_triangle(canvas : & mut MatrixXf, p1: &Vector2f, p2: &Vector2f, p3: &Vector2f) 
{
    draw_line(canvas, p1, p2);
    draw_line(canvas, p2, p3);
    draw_line(canvas, p1, p3);
}

pub fn fill_triangle(canvas: & mut MatrixXf, points: &[Vector2f; 3], val: f32)
{
    //assert!(false);
    let x1 = points.iter().cloned().map(|p| p.x()).fold1(f32::min).unwrap();
    let y1 = points.iter().cloned().map(|p| p.y()).fold1(f32::min).unwrap();
    let x2 = points.iter().cloned().map(|p| p.x()).fold1(f32::max).unwrap();
    let y2 = points.iter().cloned().map(|p| p.y()).fold1(f32::max).unwrap();
    let (i1, j1) = restrict_vector2f(canvas, &vec2f![x1, y1]);
    let (i2, j2) = restrict_vector2f(canvas, &vec2f![x2, y2]);
    for i in (i1 .. i2){
        for j in (j1 .. j2){
            let point = Vector2f{values: [i as f32, j as f32]};
            if is_point_inside_triangle(
                compute_barycentric_coords2d(points, point)){
                draw_pointf(canvas, point, val);
            }
        }
    }
}

pub fn fill_triangle_by_normals(
    canvas:  & mut MatrixXf,
    points:  &[Vector2f; 3],
    normals: &[Vector3f; 3])
{
    let oz = Vector3f{values: [0.0, 0.0, 1.0]};
    let x1 = points.iter().cloned().map(|p| p.x()).fold1(f32::min).unwrap();
    let y1 = points.iter().cloned().map(|p| p.y()).fold1(f32::min).unwrap();
    let x2 = points.iter().cloned().map(|p| p.x()).fold1(f32::max).unwrap();
    let y2 = points.iter().cloned().map(|p| p.y()).fold1(f32::max).unwrap();
    let (i1, j1) = restrict_vector2f(canvas, &vec2f![x1, y1]);
    let (i2, j2) = restrict_vector2f(canvas, &vec2f![x2, y2]);
    for i in (i1 .. i2){
        for j in (j1 .. j2){
            let point  = Vector2f{values: [i as f32, j as f32]};
            let coords = compute_barycentric_coords2d(points, point);
            let (u, v, l) = coords;
            if is_point_inside_triangle(coords){
                let normal = u * normals[0] + v * normals[1] + l * normals[2];
                let brightenss = normal.dot(&oz);
                if brightenss > 0.0 {
                    draw_pointf(canvas, point, 255.0 * brightenss);
                }
            }
        }
    }
}

pub fn fill_triangle_by_normals_and_texture(
    canvas  : & mut MatrixXf,
    texture : & MatrixXf,
    points  : &[Vector2f; 3],
    tex_verts: &[Vector2f; 3],
    normals : &[Vector3f; 3])
{
    let oz = Vector3f{values: [0.0, 0.0, 1.0]};
    let x1 = points.iter().cloned().map(|p| p.x()).fold1(f32::min).unwrap();
    let y1 = points.iter().cloned().map(|p| p.y()).fold1(f32::min).unwrap();
    let x2 = points.iter().cloned().map(|p| p.x()).fold1(f32::max).unwrap();
    let y2 = points.iter().cloned().map(|p| p.y()).fold1(f32::max).unwrap();
    let (i1, j1) = restrict_vector2f(canvas, &vec2f![x1, y1]);
    let (i2, j2) = restrict_vector2f(canvas, &vec2f![x2, y2]);
    for i in (i1 .. i2){
        for j in (j1 .. j2){
            let point  = vec2f![i as f32, j as f32];
            let coords = compute_barycentric_coords2d(points, point);
            let (u, v, l) = coords;
            if is_point_inside_triangle(coords){
                let normal   = u * normals[0]  + v * normals[1]  + l * normals[2];
                let tex_coord = u * tex_verts[0] + v * tex_verts[1] + l * tex_verts[2];
                let tex_val = texture[(
                    ((1.0 - tex_coord.y()) * texture.rows() as f32) as usize,
                    (tex_coord.x() * texture.cols() as f32) as usize)];

                let brightenss = normal.dot(&oz);
                if brightenss > 0.0 {
                    draw_pointf(canvas, point, tex_val * brightenss);
                }
            }
        }
    }
}

pub fn fill_triangle_by_normals_and_texture_rgb(
    canvas  : & mut [MatrixXf;3],
    texture : &[MatrixXf;3],
    points  : &[Vector2f; 3],
    tex_verts: &[Vector2f; 3],
    normals : &[Vector3f; 3])
{
    let oz = Vector3f{values: [0.0, 0.0, 1.0]};
    let x1 = points.iter().cloned().map(|p| p.x()).fold1(f32::min).unwrap();
    let y1 = points.iter().cloned().map(|p| p.y()).fold1(f32::min).unwrap();
    let x2 = points.iter().cloned().map(|p| p.x()).fold1(f32::max).unwrap();
    let y2 = points.iter().cloned().map(|p| p.y()).fold1(f32::max).unwrap();
    let (i1, j1) = restrict_vector2f(&canvas[0], &vec2f![x1, y1]);
    let (i2, j2) = restrict_vector2f(&canvas[0], &vec2f![x2, y2]);
    for i in (i1 .. i2){
        for j in (j1 .. j2){
            let point  = vec2f![i as f32, j as f32];
            let coords = compute_barycentric_coords2d(points, point);
            let (u, v, l) = coords;
            if is_point_inside_triangle(coords){
                let normal   = u * normals[0]  + v * normals[1]  + l * normals[2];
                let tex_coord = u * tex_verts[0] + v * tex_verts[1] + l * tex_verts[2];
                let tex_coord_plane = (
                    ((1.0 - tex_coord.y()) * texture[0].rows() as f32) as usize,
                    (tex_coord.x() * texture[0].cols() as f32) as usize);
                let texval_r = texture[0][tex_coord_plane];
                let texval_g = texture[1][tex_coord_plane];
                let texval_b = texture[2][tex_coord_plane];

                let brightenss = normal.dot(&oz);
                if brightenss > 0.0 {
                    draw_pointf(& mut canvas[0], point, texval_r * brightenss);
                    draw_pointf(& mut canvas[1], point, texval_g * brightenss);
                    draw_pointf(& mut canvas[2], point, texval_b * brightenss);
                }
            }
        }
    }
}

pub fn fill_triangle_by_normals_texture_rgb_z_buffer(
    canvas  : & mut [MatrixXf;3],
    texture : &[MatrixXf;3],
    zbuffer : & mut MatrixXf,
    points  : &[Vector2f; 3],
    depths  : &[f32; 3],
    tex_verts: &[Vector2f; 3],
    normals : &[Vector3f; 3])
{
    let oz = vec3f![0.0, 0.0, 1.0];
    let stare = oz;
    //let light_dir = oz;
    let light_dir = (vec3f![0.5, 0.0, 1.0]).normalized();
    let x1 = points.iter().cloned().map(|p| p.x()).fold1(f32::min).unwrap();
    let y1 = points.iter().cloned().map(|p| p.y()).fold1(f32::min).unwrap();
    let x2 = points.iter().cloned().map(|p| p.x()).fold1(f32::max).unwrap();
    let y2 = points.iter().cloned().map(|p| p.y()).fold1(f32::max).unwrap();
    let (i1, j1) = restrict_vector2f(&canvas[0], &vec2f![x1, y1]);
    let (i2, j2) = restrict_vector2f(&canvas[0], &vec2f![x2, y2]);
    for i in (i1 .. i2){
        for j in (j1 .. j2){
            let point  = vec2f![i as f32, j as f32];
            let coords = compute_barycentric_coords2d(points, point);
            let (u, v, l) = coords;
            if is_point_inside_triangle(coords){
                let depth    = u * depths[0] + v * depths[1] + l * depths[2];
                let normal   = u * normals[0]  + v * normals[1]  + l * normals[2];
                let tex_coord = u * tex_verts[0] + v * tex_verts[1] + l * tex_verts[2];
                let tex_coord_plane = (
                    ((1.0 - tex_coord.y()) * texture[0].rows() as f32) as usize,
                    (tex_coord.x() * texture[0].cols() as f32) as usize);
                let texval_r = texture[0][tex_coord_plane];
                let texval_g = texture[1][tex_coord_plane];
                let texval_b = texture[2][tex_coord_plane];

                let brightenss = normal.dot(&light_dir);
                let r_vec = (2.0*normal.dot(&light_dir) * normal - light_dir).normalized();
                let spec = f32::max(r_vec.dot(&stare), 0.0);
                let spec_const = 1.1;
                if 
                   brightenss > 0.0 &&
                   zbuffer[(j, i)] <= depth {
                    zbuffer[(j, i)] = depth;
                    draw_pointf(& mut canvas[0], point, 
                                f32::min(texval_r * (brightenss + spec_const*spec), 255.0));
                    draw_pointf(& mut canvas[1], point, 
                                f32::min(texval_g * (brightenss + spec_const*spec), 255.0));
                    draw_pointf(& mut canvas[2], point, 
                                f32::min(texval_b * (brightenss + spec_const*spec), 255.0));
                }
            }
        }
    }
}




//need to be rewrited
pub fn fill_triangle_by_normals_and_zbuffer(
    canvas : & mut MatrixXf,
    zbuffer: & mut MatrixXf,
    points : &[Vector2f; 3],
    depths : &[f32; 3],
    normals: &[Vector3f; 3])
{
    let oz = vec3f![0.0, 0.0, 1.0];
    let x1 = points.iter().cloned().map(|p| p.x()).fold1(f32::min).unwrap();
    let y1 = points.iter().cloned().map(|p| p.y()).fold1(f32::min).unwrap();
    let x2 = points.iter().cloned().map(|p| p.x()).fold1(f32::max).unwrap();
    let y2 = points.iter().cloned().map(|p| p.y()).fold1(f32::max).unwrap();
    let (i1, j1) = restrict_vector2f(canvas, &vec2f![x1, y1]);
    let (i2, j2) = restrict_vector2f(canvas, &vec2f![x2, y2]);
    for i in (i1 .. i2){
        for j in (j1 .. j2){
            let point  = vec2f![i as f32, j as f32];
            let coords = compute_barycentric_coords2d(points, point);
            let (u, v, l) = coords;
            if is_point_inside_triangle(coords){
                let depth  = u * depths[0] + v * depths[1] + l * depths[2];
                let normal = u * normals[0] + v * normals[1] + l * normals[2];
                let brightenss = normal.dot(&oz);
                if brightenss > 0.0
                    && zbuffer[(j, i)] <= depth 
                {
                    draw_pointf(canvas, point, 255.0 * brightenss);
                    zbuffer[(j, i)] = depth;
                }
            }
        }
    }
}




