# SCRFD - Rust Package for Face Detection

**SCRFD** is a Rust library for face detection, providing both **synchronous** and **asynchronous** support. It utilizes ONNX Runtime for high-performance inference and supports bounding box and keypoint detection.

---

## Features
- **Face Detection**: Detect bounding boxes and landmarks for faces in images.
- **Asynchronous Support**: Optional async functionality for non-blocking operations.
- **Customizable Parameters**:
  - Input size
  - Confidence threshold
  - IoU threshold
- **Efficient Processing**:
  - Anchor-based detection
  - Optimized caching for anchor centers

---

## Installation

Add the library to your `Cargo.toml`:
```toml
[dependencies]
rusty_scrfd = { version = "1.1.0", features = ["async"] } # Enable async feature if needed
```

To enable synchronous mode only, omit the `async` feature:
```toml
[dependencies]
rusty_scrfd = "1.1.0"
```

---

## Usage

### Synchronous Example
```rust
use rusty_scrfd::SCRFD;
use image::open;
use ort::session::SessionBuilder;
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load the ONNX model
    let model_path = "path/to/scrfd_model.onnx";
    let session = SessionBuilder::new().unwrap().with_model_from_file(model_path)?;

    // Initialize SCRFD
    let mut scrfd = SCRFD::new(session, (640, 640), 0.5, 0.4)?;

    // Load an image
    let image = open("path/to/image.jpg")?.into_rgb8();

    // Center cache to optimize anchor generation
    let mut center_cache = HashMap::new();

    // Detect faces
    let (bboxes, keypoints) = scrfd.detect(&image, 5, "max", &mut center_cache)?;

    println!("Bounding boxes: {:?}", bboxes);
    if let Some(kps) = keypoints {
        println!("Keypoints: {:?}", kps);
    }

    Ok(())
}
```

---

### Asynchronous Example
Enable the `async` feature in `Cargo.toml`:
```toml
[dependencies]
rusty_scrfd = { version = "1.1.0", features = ["async"] }
```

```rust
use rusty_scrfd::SCRFDAsync;
use image::open;
use ort::session::SessionBuilder;
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load the ONNX model
    let model_path = "path/to/scrfd_model.onnx";
    let session = SessionBuilder::new().unwrap().with_model_from_file(model_path)?;

    // Initialize SCRFDAsync
    let scrfd = SCRFDAsync::new((640, 640), 0.5, 0.4, session)?;

    // Load an image
    let image = open("path/to/image.jpg")?.into_rgb8();

    // Center cache to optimize anchor generation
    let mut center_cache = HashMap::new();

    // Detect faces asynchronously
    let (bboxes, keypoints) = scrfd.detect(&image, 5, "max", &mut center_cache).await?;

    println!("Bounding boxes: {:?}", bboxes);
    if let Some(kps) = keypoints {
        println!("Keypoints: {:?}", kps);
    }

    Ok(())
}
```

---

## API Documentation

### **Structs**
#### 1. `SCRFD` (Synchronous)
- **Constructor**:
  ```rust
  pub fn new(
      session: Session,
      input_size: (i32, i32),
      conf_thres: f32,
      iou_thres: f32,
  ) -> Result<Self, Box<dyn Error>>;
  ```
  - `session`: ONNX Runtime session.
  - `input_size`: Tuple of input width and height.
  - `conf_thres`: Confidence threshold for face detection.
  - `iou_thres`: IoU threshold for non-maximum suppression.

- **Methods**:
  - `detect`:
    ```rust
    pub fn detect(
        &mut self,
        image: &RgbImage,
        max_num: usize,
        metric: &str,
        center_cache: &mut HashMap<(i32, i32, i32), Array2<f32>>,
    ) -> Result<(Array2<f32>, Option<Array3<f32>>), Box<dyn Error>>;
    ```
    - Detect faces in the input image.
    - **Returns**: Detected bounding boxes and optional keypoints.

#### 2. `SCRFDAsync` (Asynchronous)
- **Constructor**:
  ```rust
  pub fn new(
      input_size: (i32, i32),
      conf_thres: f32,
      iou_thres: f32,
      session: Session,
  ) -> Result<Self, Box<dyn Error>>;
  ```
  - Same parameters as `SCRFD`.

- **Methods**:
  - `detect`:
    ```rust
    pub async fn detect(
        &self,
        image: &RgbImage,
        max_num: usize,
        metric: &str,
        center_cache: &mut HashMap<(i32, i32, i32), Array2<f32>>,
    ) -> Result<(Array2<f32>, Option<Array3<f32>>), Box<dyn Error>>;
    ```
    - Asynchronous version of `detect`.

---

### Helper Functions
**Available in `ScrfdHelpers`**:
- `generate_anchor_centers`: Efficiently generate anchor centers for feature maps.
- `distance2bbox`: Convert distances to bounding boxes.
- `distance2kps`: Convert distances to keypoints.
- `nms`: Perform non-maximum suppression to filter detections.

---

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for improvements.

### Running Tests
- For synchronous features:
  ```bash
  cargo test
  ```
- For asynchronous features:
  ```bash
  cargo test --features async
  ```

---

## License
This library is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## References
- [SCRFD Paper](https://arxiv.org/abs/2105.04714)
- [ONNX Runtime](https://onnxruntime.ai/)

