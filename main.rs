use rocket::form::{Form, DataField, FromFormField};
use rocket::serde::Deserialize;
use rocket_contrib::json::Json;
//use rocket::http::{ContentType, CookieJar, Status};

use std::fs::File;

#[macro_use] extern crate rocket;

#[derive(Deserialize)]
struct Image {
    image: Bytes,
    timestamp: String,
}

#[post("/", format = "multipart/form-data", data = "<image>")]
fn index(image: Json<Image>) -> String {
    // perform computer vision
    let bounding_box:Vec<f32> = get_bounding_box(image);
    
    // Get the necessary instructions
    // get_instructions(bounding_box)
    format!("{},{}", (0.5 - &bounding_box[0]), (0.5 - &bounding_box[1]))
}

fn get_bounding_box(coordinates: Image) -> Vec<f32> {
    // realistically what will happen here is one of two things:
    //  - use a locally trained CV model
    //  - make request for more instructions from remote CV model API.
    // either way what shall be returned here is a bounding box-
    // that returns the "coordinates" to the person that is nearest to-
    // the centre of the screen.
    // Note: bounding box returned as: 
    // <class_id> <x_centre> <y_centre> <width> <height>
    // We can forget about class_id as we will only be dealing with one:
    //  - person
    // going to assume we have obtained the values into a vector:
    vec![0.451512, 0.124531, 0.314123, 0.532123]
}

fn get_instructions(bounding_box: Vec<f32>) -> () {
    // Return value should be 2 floats denoting how much to move
    // in the x and y directions.
}

#[launch]
fn rocket() -> _ {
    rocket::build().mount("/", routes![index])
}
