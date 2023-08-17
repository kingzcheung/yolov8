use std::{ path::{ Path, PathBuf }, io::Write };

use yolov8::onnx::YOLOv8;


async fn download_models() -> (PathBuf, PathBuf, PathBuf) {
    let ptfile_url = "https://github.com/kingzcheung/yolov8/releases/download/v0.0.0/yolov8s.pt";
    let onnxfile_url =
        "https://github.com/kingzcheung/yolov8/releases/download/v0.0.0/yolov8s.onnx";
    let ttffile_url =
        "https://github.com/kingzcheung/yolov8/releases/download/v0.0.0/OpenSans-Regular.ttf";
    let testdata_dir = std::env::current_dir().unwrap();
    let testdata_dir = testdata_dir.join("../testdata");

    let ptfile = testdata_dir.join("yolov8s.pt");
    let onnxfile = testdata_dir.join("yolov8s.onnx");
    let ttffile = testdata_dir.join("OpenSans-Regular.ttf");

    if !ptfile.is_file() {
        down_url(ptfile_url, ptfile.as_path()).await;
    }
    if !onnxfile.is_file() {
        down_url(onnxfile_url, onnxfile.as_path()).await;
    }
    if !ttffile.is_file() {
        down_url(ttffile_url, ttffile.as_path()).await;
    }
    (ptfile, onnxfile, ttffile)
}


async fn down_url(url: &str, save_name: &Path) {
    let byte = reqwest::get(url).await.unwrap().bytes().await.unwrap();

    let mut file = std::fs::File::create(save_name).unwrap();
    file.write_all(&byte).unwrap();
}

#[tokio::test]
async fn predict() {
    let (_ptfile, onnxfile, _ttffile) = download_models().await;
    println!("下载完成");
    let yolo: YOLOv8 = YOLOv8::new(onnxfile).unwrap();
    let img = include_bytes!("../testdata/testssss.jpg");
    let image = image::load_from_memory_with_format(img, image::ImageFormat::Jpeg).unwrap();
    let res = yolo.predict(image).unwrap();
    dbg!(res);
}
