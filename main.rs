use std::io::Cursor as Cursor;
use image::io::Reader as ImageReader;

#[macro_use] extern crate rocket;

#[post("/")]
fn index(image: Vec<u8>)-> &'static Vec<u8> {
    // get image
    //  - and do stuff!
    
    // do computer vision
    let bounding_box: Vec<u8> = computer_vision();
    
    // get centre of person and return
    let x1 = 1; let x2 = 2;
    let y1 = 3; let y2 = 4;
    vec![round_down(x1 + x2 / 2), round_down(y1 + y2 / 2)]
}

fn computer_vision() -> Vec<u8> {
    vec![1, 2, 3]
}

#[launch]
fn rocket() -> _ {
    rocket::build().mount("/", routes![index])
}
