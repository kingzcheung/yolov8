#[derive(Debug, Clone, Copy)]
pub struct Bbox {
    pub xmin: f64,
    pub ymin: f64,
    pub xmax: f64,
    pub ymax: f64,
    pub confidence: f64,
    pub cls_index: i64,
}

impl Bbox {
    pub fn name(&self, names: &[String]) -> String {
        names[self.cls_index as usize].clone()
    }
}

// Function calculates "Intersection-over-union" coefficient for specified two boxes
// https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/.
// Returns Intersection over union ratio as a float number
pub fn iou(box1: &Bbox, box2: &Bbox) -> f64 {
    intersection(box1, box2) / union(box1, box2)
}

// Function calculates union area of two boxes
// Returns Area of the boxes union as a float number
pub fn union(box1: &Bbox, box2: &Bbox) -> f64 {
    let Bbox {
        xmin: box1_x1,
        ymin: box1_y1,
        xmax: box1_x2,
        ymax: box1_y2,
        cls_index: _,
        confidence: _,
    } = *box1;
    let Bbox {
        xmin: box2_x1,
        ymin: box2_y1,
        xmax: box2_x2,
        ymax: box2_y2,
        cls_index: _,
        confidence: _,
    } = *box2;
    let box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1);
    let box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1);
    box1_area + box2_area - intersection(box1, box2)
}

// Function calculates intersection area of two boxes
// Returns Area of intersection of the boxes as a float number
pub fn intersection(box1: &Bbox, box2: &Bbox) -> f64 {
    let Bbox {
        xmin: box1_x1,
        ymin: box1_y1,
        xmax: box1_x2,
        ymax: box1_y2,
        cls_index: _,
        confidence: _,
    } = *box1;
    let Bbox {
        xmin: box2_x1,
        ymin: box2_y1,
        xmax: box2_x2,
        ymax: box2_y2,
        cls_index: _,
        confidence: _,
    } = *box2;
    let x1 = box1_x1.max(box2_x1);
    let y1 = box1_y1.max(box2_y1);
    let x2 = box1_x2.min(box2_x2);
    let y2 = box1_y2.min(box2_y2);
    (x2 - x1) * (y2 - y1)
}
