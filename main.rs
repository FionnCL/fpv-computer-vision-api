use std::io::Cursor as Cursor;
use image::io::Reader as ImageReader;
use rocket::form::Form;

#[derive(FromForm)]
struct Task {
    complete: bool,
    description: String
}

#[macro_use] extern crate rocket;

#[post("/", data = "<task>")]
fn index(task: Form<Task>)-> &'static str {
    // do computer vision
    let bounding_box:Vec<u8> = computerVision();
    
    // Get the necessary instructions
    let instructions: str = getInstructions(&bounding_box);

    // get centre of person and return
    &instructions
}

fn computerVision() -> Vec<u8> {
    vec![1, 2, 3]
    // realistically what will happen here is one of two things:
    //  - use a locally trained CV model
    //  - make request for more instructions from remote CV model API.
    // either way what shall be returned here is a bounding box-
    // that returns the "coordinates" to the person that is nearest to-
    // the centre of the screen.
}

fn getInstructions() -> &'static str {
    "Test"
}

#[launch]
fn rocket() -> _ {
    rocket::build().mount("/", routes![index])
}
