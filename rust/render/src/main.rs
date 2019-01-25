extern crate time;
extern crate ppmiors;
#[macro_use]
extern crate rustonum;
extern crate render;
//extern crate la;
//extern crate image_routine;

#[allow(unused_imports)]
use rustonum::MatrixXf;

#[allow(unused_imports)]
use rustonum::{
    Vector3f, 
    Vector2f,
    Vector, 
    GeometryVector
};

use render::geometry_routine::{
    smooth_normals_for_vertices,
    orth_project
};

#[allow(unused_imports)]
use render::image_routine::{
    draw_line,
    draw_triangle, 
    fill_triangle, 
    fill_triangle_by_normals,
    fill_triangle_by_normals_and_zbuffer,
    fill_triangle_by_normals_and_texture,
    fill_triangle_by_normals_and_texture_rgb,
    fill_triangle_by_normals_texture_rgb_z_buffer
};

#[allow(non_snake_case)]
fn main()
{
    //let (vertices, triangleIndices, verticesTex, triangleTexIndices) = render::geometry_routine::import_obj_triangulated("/home/daiver/coding/jff/rust/render/model.obj");
    //let (vertices, triangleIndices, verticesTex, triangleTexIndices) = render::geometry_routine::import_obj_triangulated("/home/daiver/pstorage/AlexF.obj");
    //let (vertices, triangleIndices, verticesTex, triangleTexIndices) = render::geometry_routine::import_obj_triangulated("/home/daiver/pstorage/AlexPT.obj");
    let (vertices, triangleIndices, verticesTex, triangleTexIndices) = render::geometry_routine::import_obj_triangulated("/home/daiver/pstorage/AlexFT.obj");
    //let (vertices, triangleIndices, verticesTex, triangleTexIndices) = render::geometry_routine::import_obj_triangulated("/home/daiver/pstorage/AlexST.obj");
    //let texture = ppmiors::read_ppm_p5("/home/daiver/pstorage/AlexF.pgm");
    let texture = ppmiors::read_ppm_p6("/home/daiver/pstorage/AlexF.ppm");
    println!("tex shape {} {}", texture[0].rows(), texture[1].cols());


    println!("triangleTexIndices count {}", triangleTexIndices.len());
    let normals = smooth_normals_for_vertices(&vertices, &triangleIndices);
    //let (vertices, triangleIndices) = render::geometry_routine::import_obj_triangulated("../../model.obj");
    let (bbox_left_top, bbox_size)  = render::geometry_routine::bboxFromVertices(&vertices);
    let width = 1000;
    let height = 1500;
    let mut canvas  = [
            MatrixXf::zeros(height, width),
            MatrixXf::zeros(height, width),
            MatrixXf::zeros(height, width)
        ];

    let minZ = vertices.iter().cloned().map(|x| x.z()).fold(vertices[0].z(), f32::min);
    let mut zbuffer = MatrixXf::consts(height, width, minZ);
    println!("size {} {:?}i {:?}", vertices.len(), bbox_left_top, bbox_size);
    let startTime = time::now();
    println!("Start render");

    let cam_dist = 1000.0;

    for i in (0..triangleIndices.len() / 3){
        let p1 = vertices[triangleIndices[3*i + 0]];// - bbox_left_top;
        let p2 = vertices[triangleIndices[3*i + 1]];// - bbox_left_top;
        let p3 = vertices[triangleIndices[3*i + 2]];// - bbox_left_top;

        let n1 = normals[triangleIndices[3*i + 0]];
        let n2 = normals[triangleIndices[3*i + 1]];
        let n3 = normals[triangleIndices[3*i + 2]];

        let vt1 = verticesTex[triangleTexIndices[3*i + 0]];
        let vt2 = verticesTex[triangleTexIndices[3*i + 1]];
        let vt3 = verticesTex[triangleTexIndices[3*i + 2]];

        let p1 = p1 - vec3f![0.0, 0.0, bbox_size.z()];
        let p2 = p2 - vec3f![0.0, 0.0, bbox_size.z()];
        let p3 = p3 - vec3f![0.0, 0.0, bbox_size.z()];

        let p1 = orth_project(&p1, cam_dist);
        let p2 = orth_project(&p2, cam_dist);
        let p3 = orth_project(&p3, cam_dist);

        let p1 = p1 - bbox_left_top;
        let p2 = p2 - bbox_left_top;
        let p3 = p3 - bbox_left_top;

        //println!("{:?} {:?} {:?}", vt1, vt2, vt3);

//        fill_triangle_by_normals(
                     //&mut canvas, 
                     //&[
                         //Vector2f {y: 700.0 - p1.y.round(), x:p1.x.round()},
                         //Vector2f {y: 700.0 - p2.y.round(), x:p2.x.round()},
                         //Vector2f {y: 700.0 - p3.y.round(), x:p3.x.round()},
                     //],
                     //&[n1, n2, n3]
                                //);
//        fill_triangle_by_normals_and_zbuffer(
                     //&mut canvas, 
                     //&mut zbuffer,
                     //&[
                         //Vector2f {values: [p1.x().round(), 700.0 - p1.y().round()]},
                         //Vector2f {values: [p2.x().round(), 700.0 - p2.y().round()]},
                         //Vector2f {values: [p3.x().round(), 700.0 - p3.y().round()]},
                     //],
                     //&[p1.z(), p2.z(), p3.z()],
                     //&[n1, n2, n3]);
        fill_triangle_by_normals_texture_rgb_z_buffer(
                     &mut canvas, 
                     & texture,
                     & mut zbuffer,
                     &[
                         Vector2f {values: [p1.x().round(), 700.0 - p1.y().round()]}*2.0,
                         Vector2f {values: [p2.x().round(), 700.0 - p2.y().round()]}*2.0,
                         Vector2f {values: [p3.x().round(), 700.0 - p3.y().round()]}*2.0,
                     ],
                     &[p1.z(), p2.z(), p3.z()],
                     &[vt1, vt2, vt3],
                     &[n1, n2, n3]);

//        fill_triangle_by_normals_and_texture_rgb(
                     //&mut canvas, 
                     //& texture,
                     //&[
                         //Vector2f {values: [p1.x().round(), 700.0 - p1.y().round()]}*2.0,
                         //Vector2f {values: [p2.x().round(), 700.0 - p2.y().round()]}*2.0,
                         //Vector2f {values: [p3.x().round(), 700.0 - p3.y().round()]}*2.0,
                     //],
                     //&[vt1, vt2, vt3],
                     //&[n1, n2, n3]);
    }
    let finishTime = time::now();
    println!("Finish render");
    println!("elapsed {}", finishTime - startTime);
    ppmiors::write_ppm_p6(&canvas, "tmp.ppm");
    let zmin    = zbuffer.min();
    let zmax    = zbuffer.max();
    zbuffer = 255.0 * ((-zmin + zbuffer)*(1.0/(zmax - zmin)));
    ppmiors::write_ppm_p5(&zbuffer, "z.ppm");
    //ppmiors::save_ppm_p2(&texture, "tex.ppm");
}
