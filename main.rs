//use rocket::form::{Form, DataField, FromFormField};
//use rocket::http::{ContentType, CookieJar, Status};

use std::{
    env,
    error::Error,
    io,
    ffi::OsString,
    fs::File,
    process
};

#[macro_use] extern crate rocket;

#[post("/")]
fn index(task: File)-> &'static str{
    // do computer vision
    let bounding_box:Vec<u8> = computer_vision();
    
    // Get the necessary instructions
    // get_instructions(bounding_box)
    (std::fs::read_to_string(task)).as_str()
}

fn computer_vision(coordinates: File) -> Vec<f32> {
    let mut deconstructed_values = run().iter().position(|x| * x == 0).unwrap(); 

    vec![0.451512, 0.124531, 0.314123, 0.532123]
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
}

fn get_instructions(bounding_box: Vec<f32>) -> &'static str {
    "Test"
}

fn run() -> Result<(), Box<dyn Error>> {
    let file_path = get_first_arg()?;
    let file = File::open(file_path)?;
    let mut rdr = csv::Reader::new()
        .has_headers(false)
        .delimiter(b' ')
        .from_reader(file);
    for result in rdr.records() {
        let record = result?;
        println!("{:?}", record);
    }
    Ok(())
}

/// Returns the first positional argument sent to this process. If there are no
/// positional arguments, then this returns an error.
fn get_first_arg() -> Result<OsString, Box<dyn Error>> {
    match env::args_os().nth(1) {
        None => Err(From::from("expected 1 argument, but got none")),
        Some(file_path) => Ok(file_path),
    }
}

#[launch]
fn rocket() -> _ {
    rocket::build().mount("/", routes![index])
}
