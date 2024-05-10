use std::io::Cursor;
use image::io::Reader as ImageReader;

#[macro_use] extern crate rocket;

#[post("/")]
fn index(image: Vec<u8>)-> &'static Vec<u8> {
    // get image
    //  - and do stuff!
    
    // do computer vision
    let bounding_box: Vec<u8> = computerVision();
    
    // get centre of person and return
    vec![((bounding_box[x1] + bounding_box[x2]) / 2), ((bounding_box[y1] + bounding_box[y2]) / 2)]
}

#[launch]
fn rocket() -> _ {
    rocket::build().mount("/", routes![index])
}
