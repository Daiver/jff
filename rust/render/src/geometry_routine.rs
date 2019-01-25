use std;
use std::io::prelude::*;
use std::fs::File;

use rustonum::{Vector2f, Vector3f, GeometryVector, LAObject};

pub fn orth_project(point: &Vector3f, cam_dist: f32) -> Vector3f
{
    let denominator = 1.0 - point.z()/cam_dist;
    vec3f![point.x()/denominator, point.y()/denominator, point.z()/denominator]
}

pub fn compute_barycentric_coords2d(points: &[Vector2f; 3], p: Vector2f) -> (f32, f32, f32)
{
    let p1 = points[0];
    let p2 = points[1];
    let p3 = points[2];
    let denominator = 
        (p2.y() - p3.y()) * (p1.x() - p3.x()) + (p3.x() - p2.x()) * (p1.y() - p3.y());

    let lambda1 = (
            (p2.y() - p3.y()) * (p.x() - p3.x()) + (p3.x() - p2.x()) * (p.y() - p3.y())
        ) / denominator;
    let lambda2 = (
            (p3.y() - p1.y()) * (p.x() - p3.x()) + (p1.x() - p3.x()) * (p.y() - p3.y())
        ) / denominator;
    let lambda3 = 1.0 - lambda1 - lambda2;
    (lambda1, lambda2, lambda3)
}

pub fn is_point_inside_triangle(coords: (f32, f32, f32)) -> bool
{
    let (u, v, l) = coords;
    (
        (u + v + l - 1.0).abs() < 0.001 &&
        u >= -0.0001 && u <= 1.0001 &&
        v >= -0.0001 && v <= 1.0001 &&
        l >= -0.0001 && l <= 1.0001 
    )
}

pub fn normal_for_triangle(points: &[Vector3f ; 3]) -> Vector3f
{
    let a = points[1] - points[0];
    let b = points[2] - points[0];
    a.cross(b).normalized()
}

#[allow(non_snake_case)]
pub fn smooth_normals_for_vertices(
    vertices:        &Vec<Vector3f>, 
    triangle_indices: &Vec<usize>) -> Vec<Vector3f>
{
    let nPoints = (triangle_indices.len() / 3) as usize;
    let mut normals: Vec<Vector3f> = Vec::with_capacity(nPoints);
    for triInd in (0 .. nPoints){
        let p1 = vertices[triangle_indices[3*triInd + 0]];
        let p2 = vertices[triangle_indices[3*triInd + 1]];
        let p3 = vertices[triangle_indices[3*triInd + 2]];
        normals.push(normal_for_triangle(&[p1, p2, p3]));
    }

    let mut res: Vec<Vector3f> = std::iter::repeat(Vector3f{values: [0.0, 0.0, 0.0]})
                                    .take(vertices.len()).collect::<Vec<_>>();

    for i in (0 .. triangle_indices.len()){
        let ind = triangle_indices[i];
        let triInd = i / 3;
        res[ind] = res[ind] + normals[triInd];
    }
    res.iter().map(|x| x.normalized()).collect::<Vec<_>>()
}

pub fn import_obj_triangulated(fname : &str) -> 
    (Vec<Vector3f>, Vec<usize>, Vec<Vector2f>, Vec<usize>)
{
    let mut f = File::open(fname).unwrap();
    let mut raw_data = String::new();
    f.read_to_string(&mut raw_data).unwrap();

    let mut vertices           : Vec<Vector3f> = Vec::new();
    let mut triangle_indices    : Vec<usize>    = Vec::new();

    let mut vertices_tex        : Vec<Vector2f> = Vec::new();
    let mut triangle_tex_indices : Vec<usize>    = Vec::new();

    let splt = raw_data.split("\n").filter(|&x| x.len() > 0);
    for s in splt {
        let mut tokens = s.split(" ");
        let token_head = match tokens.next() {
            None => "",
            Some(x) => x
        };

        let is_tex_used = tokens.clone().filter(|&x| x.len() > 0).next().unwrap().to_string().split("//").count() == 1;

        match token_head {
            "v" => {
                let coords = tokens.take(3)
                    .map(|x| x.parse::<f32>().unwrap()).collect::<Vec<_>>();
                let x = vec3f![coords[0], coords[1], coords[2]];
                vertices.push(x);
            },
            "vt" => {
                let coords = tokens.take(2)
                    .map(|x| x.parse::<f32>().unwrap()).collect::<Vec<_>>();
                let x = vec2f![coords[0], coords[1]];
                vertices_tex.push(x);
            },
            "f" =>{
                let face_items = tokens.filter(|&x| x.len() > 0).take(3);
                let indices = face_items.map(|x| x.split("/").filter(|&x| x.len() > 0)
                                       .map(|x| x.parse::<u32>().unwrap()).collect::<Vec<_>>());
                for inds in indices{
                    let i = inds[0];
                    triangle_indices.push((i - 1) as usize);
                    if is_tex_used && inds.len() > 1 {
                        triangle_tex_indices.push((inds[1] - 1) as usize);
                    }
                }
            },
            _ => ()
        }
    }
    (vertices, triangle_indices, vertices_tex, triangle_tex_indices)
}

#[allow(non_snake_case)]
pub fn bboxFromVertices(vertices: &Vec<Vector3f>) -> (Vector3f, Vector3f)
{
    let mut leftUp    = vertices[0];
    let mut rightDown = vertices[0];

    for i in (0..vertices.len()){
        let &v = &vertices[i];
        for j in (0 .. v.size()){
            if leftUp[j] > v[j] {
                leftUp[j] = v[j];
            }
            if rightDown[j] < v[j] {
                rightDown[j] = v[j];
            }
        }
    }

    (leftUp, rightDown - leftUp)
}


