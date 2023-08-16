use std::{path::Path, time::Instant};

use image::{Rgb, ImageBuffer};
use imageproc::rect::Rect;
use rusttype::{Scale, Font};
use tch::{nn::ModuleT, Device, IndexOp, Tensor, TchError};

use crate::bbox::Bbox;

#[derive(Debug)]
pub struct YOLOv8 {
    model: tch::CModule,
    pub device: tch::Device,
    h: i64,
    w: i64,
    conf_threshold: f64,
    iou_threshold: f64,
    top_k: i64,
}

impl YOLOv8 {
    pub fn new(
        weights: &Path,
        h: i64,
        w: i64,
        conf_threshold: f64,
        iou_threshold: f64,
        top_k: i64,
        device: Device,
    ) -> Result<YOLOv8,TchError> {
        let mut model = tch::CModule::load_on_device(weights, device)?;
        model.set_eval();
        Ok(
            YOLOv8 {
                model,
                device,
                h,
                w,
                conf_threshold,
                iou_threshold,
                top_k,
            }
        )
        
    }

    pub fn preprocess(&mut self, image_path: &str) -> Result<Tensor,TchError> {
        let origin_image = tch::vision::image::load(image_path)?;
        let (_, ori_h, ori_w) = origin_image.size3()?;
        self.w = ori_w;
        self.h = ori_h;
        let img = tch::vision::image::resize(&origin_image, 640, 640)?
            .unsqueeze(0)
            .to_kind(tch::Kind::Float)
            / 255.;

        Ok(img)
    }

    pub fn predict(&mut self, image: &tch::Tensor) -> Vec<Bbox> {
        let start_time = Instant::now();
        let pred = self.model.forward_t(image, false);
        let elapsed_time = start_time.elapsed();
        println!("libtorch inference time:{} ms", elapsed_time.as_millis());

        let pred = pred.to_device(self.device);
        let start_time = Instant::now();
        let result = self.non_max_suppression(&pred);
        let elapsed_time = start_time.elapsed();
        println!("libtorch nms time:{} ms", elapsed_time.as_millis());
        result
    }

    fn non_max_suppression(&self, pred: &tch::Tensor) -> Vec<Bbox> {
        let pred = &pred.transpose(2, 1).squeeze();
        let (npreds, pred_size) = pred.size2().unwrap();
        let nclasses = pred_size - 4;
        let mut bboxes: Vec<Vec<Bbox>> = (0..nclasses).map(|_| vec![]).collect();

        let class_index = pred.i((.., 4..pred_size));
        let (pred_conf, class_label) = class_index.max_dim(-1, false);
        // pred_conf.save("pred_conf.pt").expect("pred_conf save err");
        // class_label.save("class_label.pt").expect("class_labe; save err");
        for index in 0..npreds {
            let pred = Vec::<f64>::try_from(pred.get(index)).unwrap();
            let conf = pred_conf.double_value(&[index]);
            if conf > self.conf_threshold {
                let label = class_label.int64_value(&[index]);
                if pred[(4 + label) as usize] > 0. {
                    let bbox = Bbox {
                        xmin: pred[0] - pred[2] / 2.,
                        ymin: pred[1] - pred[3] / 2.,
                        xmax: pred[0] + pred[2] / 2.,
                        ymax: pred[1] + pred[3] / 2.,
                        confidence: conf,
                        cls_index: label
                    };
                    bboxes[label as usize].push(bbox);
                }
            }
        }

        for bboxes_for_class in bboxes.iter_mut() {
            bboxes_for_class.sort_by(|b1, b2| b2.confidence.partial_cmp(&b1.confidence).unwrap());
            let mut current_index = 0;
            for index in 0..bboxes_for_class.len() {
                let mut drop = false;
                for prev_index in 0..current_index {
                    let iou = self.iou(&bboxes_for_class[prev_index], &bboxes_for_class[index]);
                    if iou > self.iou_threshold {
                        drop = true;
                        break;
                    }
                }
                if !drop {
                    bboxes_for_class.swap(current_index, index);
                    current_index += 1;
                }
            }
            bboxes_for_class.truncate(current_index);
        }
        let mut result = vec![];
        let mut count = 0;
        for bboxes_for_class in bboxes.iter() {
            for b in bboxes_for_class.iter() {
                if count >= self.top_k {
                    break;
                }
                result.push(*b);
                count += 1;
            }
        }

        result
    }

    fn iou(&self, b1: &Bbox, b2: &Bbox) -> f64 {
        let b1_area = (b1.xmax - b1.xmin + 1.) * (b1.ymax - b1.ymin + 1.);
        let b2_area = (b2.xmax - b2.xmin + 1.) * (b2.ymax - b2.ymin + 1.);
        let i_xmin = b1.xmin.max(b2.xmin);
        let i_xmax = b1.xmax.min(b2.xmax);
        let i_ymin = b1.ymin.max(b2.ymin);
        let i_ymax = b1.ymax.min(b2.ymax);
        let i_area = (i_xmax - i_xmin + 1.).max(0.) * (i_ymax - i_ymin + 1.).max(0.);
        i_area / (b1_area + b2_area - i_area)
    }


    pub fn show(&self, mut image: ImageBuffer<Rgb<u8>, Vec<u8>>, bboxes: &[Bbox], names:&[String]) {

        let w_ratio = self.w as f64 / 640_f64;
        let h_ratio = self.h as f64 / 640_f64;
        let green = Rgb([0u8,   255u8, 0u8]);
        for bbox in bboxes.iter() {
            let xmin = ((bbox.xmin * w_ratio) as i64).clamp(0, self.w);
            let ymin = ((bbox.ymin * h_ratio) as i64).clamp(0, self.h);
            let xmax = ((bbox.xmax * w_ratio) as i64).clamp(0, self.w);
            let ymax = ((bbox.ymax * h_ratio) as i64).clamp(0, self.h);

            let width = (xmax - xmin).unsigned_abs() as u32;
            let height = (ymax - ymin).unsigned_abs() as u32;
            let rect = Rect::at(xmin as i32, ymin as i32).of_size(width, height);
            imageproc::drawing::draw_hollow_rect_mut(&mut image, rect, green);
            let height = 20;
            let scale = Scale {
                x: height as f32 * 1.5,
                y: height as f32,
            };
            let font = Vec::from(include_bytes!("../testdata/OpenSans-Regular.ttf") as &[u8]);
            let font = Font::try_from_vec(font).unwrap();
            let label = bbox.name(names);
            imageproc::drawing::draw_text_mut(&mut image, green, xmin as i32, ymin as i32 - height, scale, &font, &label)
        }
        image.save("./result.jpg").unwrap();
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::YOLOv8;
    use tch::Tensor;

    #[test]
    pub fn cuda_is_available() {
        println!("Cuda available: {}", tch::Cuda::is_available());
        println!("Cudnn available: {}", tch::Cuda::cudnn_is_available());
        let device = tch::Device::cuda_if_available();
        println!("Device :{:?}", device);
        let t = Tensor::from_slice(&[1, 2, 3, 4, 5]).to(device);
        t.print();
    }

    #[test]
    pub fn test_yolov8() {
        let weights = "../testdata/best.torchscript";
        let weights = Path::new(weights);
        let (h, w) = (720, 720);
        let conf_threshold = 0.25;
        let iou_threshold = 0.45;
        let top_k = 100;
        let device = tch::Device::Cpu;
        let mut yolo = YOLOv8::new(weights, h, w, conf_threshold, iou_threshold, top_k, device).unwrap();
        let img = yolo.preprocess("../testdata/testssss.jpg").unwrap();
        let output = yolo.predict(&img);
        println!("output: {:?}, size: {}", output, output.len());

        let img = image::open("../testdata/testssss.jpg").unwrap();
        yolo.show(img.into_rgb8(), &output,&["object"].map(|x| x.to_string()));
    }
}