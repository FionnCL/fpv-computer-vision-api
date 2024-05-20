use image::DynamicImage;
use object_detector_rust::prelude::*;
use object_detector_rust::utils::extract_data;
use object_detector_rust::feature::HOGFeature;
// Create a memory-based DataSet
let mut dataset = MemoryDataSet::new();

// Add some annotated images to the DataSet
dataset.add_annotated_image(AnnotatedImage {
    image: DynamicImage::new_rgba8(128, 128),
    annotations: vec![Annotation {
        bbox: BBox { x: 0, y: 0, width: 32, height: 32 },
        class: 0,
    }],
});
dataset.add_annotated_image(AnnotatedImage {
    image: DynamicImage::new_rgba8(128, 128),
    annotations: vec![Annotation {
        bbox: BBox { x: 50, y: 50, width: 32, height: 32 },
        class: 1,
    }],
});
let class = 1;
let feature = HOGFeature::default();
// Create a BayesClassifier and train it on the DataSet
let mut classifier = BayesClassifier::new();
let (x, y) = dataset.get_data();
let x: Vec<Vec<f32>> = x
        .iter()
        .map(|image| feature.extract(image).unwrap())
        .collect();
    let y = y
        .iter()
        .map(|y| if *y == class { true } else { false })
        .collect();
let (x, y) = extract_data(x, y);
classifier.fit(&x.view(), &y.view());
