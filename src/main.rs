use image::DynamicImage;
use object_detector_rust::prelude::*;
use object_detector_rust::utils::extract_data;
use object_detector_rust::feature::HOGFeature;

fn main() {
    // Create a memory-based DataSet
    let mut dataset = MemoryDataSet::new();
    

}

fn add_annotated_images(){
    printf!("{}", serde_json::from_str("./labels/labels.json"));
    let parsed_json = json::parse(serde_json::from_str("./labels/labels.json"))
        .unwrap();

    
}
