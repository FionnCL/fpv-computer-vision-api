use rocket::fs::FileName;
use rocket::data::ToByteUnit;

use rocket::form::{Form, DataField, FromFormField};
use rocket::http::{ContentType, CookieJar, Status};

#[macro_use] extern crate rocket;

struct File<'v>{
    file_name: Option<&'v FileName>,
    content_type: ContentType,
    data: Vec<u8>
}

#[post("/", data = "<task>")]
fn index(task: Form<File>)-> &'static str {
    // do computer vision
    let bounding_box:Vec<u8> = computerVision();
    
    // Get the necessary instructions
    getInstructions(bounding_box)
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

fn getInstructions(bounding_box: Vec<u8>) -> &'static str {
    "Test"
}

#[rocket::async_trait]
impl<'v> FromFormField<'v> for File<'v> {
    async fn from_data(field: DataField<'v, '_>) -> rocket::form::Result<'v, Self> {
        let stream = field.data.open(u32::MAX.bytes());
        let bytes = stream.into_bytes().await?;
        Ok(File {
            file_name: field.file_name,
            content_type: field.content_type,
            data: bytes.value,
        })

    }
}

#[launch]
fn rocket() -> _ {
    rocket::build().mount("/", routes![index])
}
