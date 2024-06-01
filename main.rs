use opencv::prelude::*;
use opencv::imgcodecs;

mod yolo;

#[macro_use] extern crate rocket;

#[post("/")]
fn index() -> String {
    // perform computer vision
    //let bounding_box:Vec<f32> = get_bounding_box(image);
    
    // Get the necessary instructions
    // get_instructions(bounding_box)
    //format!("{},{}", (0.5 - &bounding_box[0]), (0.5 - &bounding_box[1]))
    String::from("lol")
}

fn get_bounding_box() -> Vec<f32> {
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

fn get_instructions(_bounding_box: Vec<f32>) -> () {
    // Return value should be 2 floats denoting how much to move
    // in the x and y directions.
}

#[launch]
fn rocket() -> _ {
    println!("Loading model...");
    let mut model = match yolo::load_model() {
        Ok(model) => model,
        Err(e) => {
            println!("Error: {}", e);
            std::process::exit(0);
        }
    };

    println!("Reading test image...");
    let img_path = "data/test-image.jpg";
    let mut img = match imgcodecs::imread(img_path, imgcodecs::IMREAD_COLOR) {
        Ok(img) => img,
        Err(e) => {
            println!("Error: {}", e);
            std::process::exit(0);
        }
    };
    if img.size().unwrap().width > 0 {
        println!("Image loaded successfully.");
    } else {
        println!("Failed to load image.");
        std::process::exit(0);
    }

    let detections = yolo::detect(&mut model, &img, 0.5, 0.5);
    if detections.is_err() {
        println!("Failed to detect, {:?}", detections.err().unwrap());
        std::process::exit(0);
    }

    let detections = detections.unwrap();
    println!("{:?}", detections);
    yolo::draw_predictions(&mut img, &detections, &model.model_config);

    let params: opencv::core::Vector<i32> = opencv::core::Vector::default();
    opencv::imgcodecs::imwrite("result.jpg", &img, &params).unwrap();

    rocket::build().mount("/", routes![index])
}
