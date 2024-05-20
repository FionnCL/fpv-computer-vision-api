use image::DynamicImage;
use crate::prelude::BBox;
use image::{imageops::crop_imm, DynamicImage, GenericImageView, Rgba};
use imageproc::{drawing::draw_hollow_rect_mut, rect::Rect};
use imageproc::hog::{hog, HogOptions};

use ndarray::{Array1, Array2};
use super::Feature;

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

/// Transforms a given vector of extracted features and a vector of labels into a tuple of 2D feature array and 1D label array.
pub fn extract_data<X, Y>(features: Vec<Vec<X>>, labels: Vec<Y>) -> (Array2<X>, Array1<Y>) {
    assert_eq!(features.len(), labels.len());
    let (x, y) = (
        labels.len(),
        match features.first() {
            Some(x) => x.len(),
            None => 0,
        },
    );
    let features = features.into_iter().flatten().collect();
    let features_array = Array2::from_shape_vec((x, y), features).unwrap();
    let labels_array = Array1::from_shape_vec(labels.len(), labels).unwrap();

    (features_array, labels_array)
}

impl Default for HOGFeature {
    /// Creates a new HOG feature extractor with default options
    fn default() -> Self {
        let default_options = HogOptions {
            orientations: 9,
            signed: false,
            cell_side: 8,
            block_side: 2,
            block_stride: 1,
        };
        Self {
            options: default_options,
        }
    }
}

impl Feature for HOGFeature {
    fn extract(&self, image: &DynamicImage) -> Result<Vec<f32>, String> {
        match image.as_luma8() {
            Some(gray_image) => hog(gray_image, self.options),
            None => hog(&image.to_luma8(), self.options),
        }
    }
}
