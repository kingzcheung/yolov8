use std::{
    path::Path,
    sync::Arc, time::Instant,
};

use crate::bbox::{iou, Bbox};
use image::{imageops::FilterType, GenericImageView, DynamicImage};
use ndarray::{s, Array, Axis, IxDyn};
use ort::{Environment, Session, SessionBuilder,Value};

pub struct YOLOv8 {
    model: Session,
}

impl YOLOv8 {
    pub fn new<P>(onnx_file: P) -> Result<YOLOv8, ort::OrtError>
    where
        P: AsRef<Path>,
    {
        let env = Arc::new(Environment::builder().with_name("YOLOv8").build()?);

        let model = SessionBuilder::new(&env)?.with_model_from_file(onnx_file)?;
        Ok(Self { model })
    }

    pub fn predict(&self, image:DynamicImage) -> Result<Vec<Bbox>, ort::OrtError> {
        let (input, img_width, img_height) = self.prepare_input(image);
        let start_time = Instant::now();
        let output = self.run_model(input)?;
        println!("onnx inference time:{} ms", start_time.elapsed().as_millis());
        let res = self.process_output(output, img_width, img_height);
        Ok(res)
    }

    fn prepare_input(&self, img: DynamicImage) -> (Array<f32, IxDyn>, u32, u32) {
        // let img: image::DynamicImage = image::load_from_memory_with_format(&buf, image::ImageFormat::Jpeg).unwrap();
        let (img_width, img_height) = (img.width(), img.height());
        let img = img.resize_exact(640, 640, FilterType::CatmullRom);
        let mut input = Array::zeros((1, 3, 640, 640)).into_dyn();
        for pixel in img.pixels() {
            let x = pixel.0 as usize;
            let y = pixel.1 as usize;
            let [r, g, b, _] = pixel.2 .0;
            input[[0, 0, y, x]] = (r as f32) / 255.0;
            input[[0, 1, y, x]] = (g as f32) / 255.0;
            input[[0, 2, y, x]] = (b as f32) / 255.0;
        }
        (input, img_width, img_height)
    }
    fn run_model(&self, input: Array<f32, IxDyn>) -> Result<Array<f32, IxDyn>, ort::OrtError> {
        let input_as_values = &input.as_standard_layout();
        let model_inputs = vec![Value::from_array(self.model.allocator(), input_as_values)?];
        let outputs = self.model.run(model_inputs)?;
        let output = outputs
            .get(0)
            .unwrap()
            .try_extract::<f32>()?
            .view()
            .t()
            .into_owned();
        Ok(output)
    }

    #[allow(clippy::manual_retain)]
    fn process_output(
        &self,
        output: Array<f32, IxDyn>,
        img_width: u32,
        img_height: u32,
    ) -> Vec<Bbox> {
        let mut boxes = Vec::new();
        let output = output.slice(s![.., .., 0]);
        for row in output.axis_iter(Axis(0)) {
            let row: Vec<_> = row.iter().copied().collect();
            let (class_id, prob) = row
                .iter()
                .skip(4)
                .enumerate()
                .map(|(index, value)| (index, *value))
                .reduce(|accum, row| if row.1 > accum.1 { row } else { accum })
                .unwrap();
            if prob < 0.5 {
                continue;
            }
            // let label = YOLO_CLASSES[class_id];
            let xc = row[0] / 640.0 * (img_width as f32);
            let yc = row[1] / 640.0 * (img_height as f32);
            let w = row[2] / 640.0 * (img_width as f32);
            let h = row[3] / 640.0 * (img_height as f32);
            let x1 = xc - w / 2.0;
            let x2 = xc + w / 2.0;
            let y1 = yc - h / 2.0;
            let y2 = yc + h / 2.0;
            // (x1,y1,x2,y2,label,prob)
            boxes.push(Bbox {
                xmin: x1 as f64,
                ymin: y1 as f64,
                xmax: x2 as f64,
                ymax: y2 as f64,
                confidence: prob as f64,
                cls_index: class_id as i64,
            });
        }

        boxes.sort_by(|box1, box2| box2.confidence.total_cmp(&box1.confidence));
        let mut result = Vec::new();
        while !boxes.is_empty() {
            result.push(boxes[0]);
            boxes = boxes
                .iter()
                .filter(|box1| iou(&boxes[0], box1) < 0.7)
                .copied()
                .collect()
        }
        result
    }
}

#[cfg(test)]
mod test {

    #[test]
    fn predict() {
        let onnx_file = "testdata/best.onnx";
        let yolo: super::YOLOv8 = super::YOLOv8::new(onnx_file).unwrap();
        let img = include_bytes!("../testdata/testssss.jpg");
        let image = image::load_from_memory_with_format(img, image::ImageFormat::Jpeg).unwrap();
        let res = yolo.predict(image).unwrap();
        dbg!(res);
    }
}
