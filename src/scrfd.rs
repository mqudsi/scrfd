use crate::helpers::ScrfdHelpers;
use image::RgbImage;
use ndarray::{s, Array2, Array3, Array4, ArrayD, ArrayViewD, Axis};
use ort::{session::Session, value::Value};
use std::{collections::HashMap, error::Error};

pub struct SCRFD {
    input_size: (i32, i32),
    conf_thres: f32,
    iou_thres: f32,

    // SCRFD model parameters
    _fmc: usize,
    feat_stride_fpn: Vec<i32>,
    num_anchors: usize,
    use_kps: bool,

    mean: f32,
    std: f32,

    session: Session,
    _output_names: Vec<String>,
    input_names: Vec<String>,
}

impl SCRFD {
    /// Constructor to initialize the SCRFD model
    /// # Arguments:
    /// - model_path: Path to the SCRFD model
    /// - input_size: Tuple of (width, height) for the input image
    /// - conf_thres: Confidence threshold
    /// - iou_thres: IoU threshold
    /// # Returns:
    /// - Self
    pub fn new(
        session: Session,
        input_size: (i32, i32),
        conf_thres: f32,
        iou_thres: f32,
    ) -> Result<Self, Box<dyn Error>> {
        // SCRFD model parameters
        let fmc = 3;
        let feat_stride_fpn = vec![8, 16, 32];
        let num_anchors = 2;
        let use_kps = true;

        let mean = 127.5;
        let std = 128.0;

        // Get model input and output names
        let output_names = session.outputs.iter().map(|o| o.name.clone()).collect();
        let input_names = session.inputs.iter().map(|i| i.name.clone()).collect();

        Ok(SCRFD {
            input_size,
            conf_thres,
            iou_thres,
            _fmc: fmc,
            feat_stride_fpn,
            num_anchors,
            use_kps,
            mean,
            std,
            session,
            _output_names: output_names,
            input_names,
        })
    }

    /// Prepare the input tensor for the model
    /// # Arguments:
    /// - image: RgbImage
    /// # Returns:
    /// - Array4<f32>
    fn prepare_input_tensor(&self, image: &RgbImage) -> Result<Array4<f32>, Box<dyn Error>> {
        // Convert image to float32
        let mut float_image = image.clone();
        for pixel in float_image.pixels_mut() {
            let [r, g, b] = pixel.0;
            // Swap R and B channels (swapRB=True)
            pixel.0 = [b, g, r];
        }

        // Create the input tensor
        let (width, height) = float_image.dimensions();
        let mut input_tensor = Array4::<f32>::zeros((1, 3, height as usize, width as usize));

        for (x, y, pixel) in float_image.enumerate_pixels() {
            let [b, g, r] = pixel.0;
            // Apply preprocessing: ((pixel_value - mean) * scalefactor)
            input_tensor[[0, 0, y as usize, x as usize]] =
                (b as f32 - self.mean) * (1.0 / self.std);
            input_tensor[[0, 1, y as usize, x as usize]] =
                (g as f32 - self.mean) * (1.0 / self.std);
            input_tensor[[0, 2, y as usize, x as usize]] =
                (r as f32 - self.mean) * (1.0 / self.std);
        }

        Ok(input_tensor)
    }

    /// The forward method processes the image and runs the model
    /// # Arguments:
    /// - input_tensor: &ArrayD<f32>
    /// # Returns:
    /// - Result<(Vec<Array2<f32>>, Vec<Array2<f32>>, Vec<Array3<f32>>), Box<dyn Error>>
    pub fn forward(
        &mut self,
        input_tensor: &ArrayD<f32>,
        center_cache: &mut HashMap<(i32, i32, i32), Array2<f32>>,
    ) -> Result<(Vec<Array2<f32>>, Vec<Array2<f32>>, Vec<Array3<f32>>), Box<dyn Error>> {
        let mut scores_list = Vec::new();
        let mut bboxes_list = Vec::new();
        let mut kpss_list = Vec::new();
        let input_height = input_tensor.shape()[2];
        let input_width = input_tensor.shape()[3];
        let input_value = Value::from_array(input_tensor.to_owned())?;

        // Run the model
        let session_output = self
            .session
            .run(ort::inputs![self.input_names[0].as_str() => input_value]?)?;
        let mut outputs = vec![];
        for (_, output) in session_output.iter().enumerate() {
            let f32_array: ArrayViewD<f32> = output.1.try_extract_tensor()?;
            outputs.push(f32_array.to_owned());
        }
        drop(session_output);

        // reverse the order of outputs
        for (idx, &stride) in self.feat_stride_fpn.iter().enumerate() {
            let base_idx = idx * 3;
            let scores = outputs[base_idx].clone();
            let bbox_preds =
                outputs[base_idx + 1].to_shape((outputs[base_idx + 1].len() / 4, 4))?;
            let bbox_preds = bbox_preds * stride as f32;
            let kps_preds = outputs[base_idx + 2]
                .to_shape((outputs[base_idx + 2].len() / 10, 10))?
                * stride as f32;

            // Determine feature map dimensions
            let height = input_height / stride as usize;
            let width = input_width / stride as usize;

            // Generate anchor centers
            let key = (height as i32, width as i32, stride);
            let anchor_centers = if let Some(centers) = center_cache.get(&key) {
                centers.clone()
            } else {
                let centers = ScrfdHelpers::generate_anchor_centers(
                    self.num_anchors,
                    height,
                    width,
                    stride as f32,
                );
                if center_cache.len() < 100 {
                    center_cache.insert(key, centers.clone());
                }
                centers
            };

            // Filter scores by threshold
            let pos_inds: Vec<usize> = scores
                .iter()
                .enumerate()
                .filter(|(_, &s)| s > self.conf_thres)
                .map(|(i, _)| i)
                .collect();

            if pos_inds.is_empty() {
                continue;
            }

            println!("Scores: {:?}", scores.shape());
            let pos_scores = scores.select(Axis(0), &pos_inds);
            let bboxes = ScrfdHelpers::distance2bbox(&anchor_centers, &bbox_preds.to_owned(), None);
            let pos_bboxes = bboxes.select(Axis(0), &pos_inds);

            println!("Pos scores: {:?}", pos_scores.shape());
            scores_list.push(pos_scores.to_shape((pos_scores.len(), 1))?.to_owned());
            bboxes_list.push(pos_bboxes);

            if self.use_kps {
                let kpss = ScrfdHelpers::distance2kps(&anchor_centers, &kps_preds.to_owned(), None);
                let kpss = kpss.to_shape((kpss.shape()[0], kpss.shape()[1] / 2, 2))?;
                let pos_kpss = kpss.select(Axis(0), &pos_inds);
                kpss_list.push(pos_kpss);
            }
        }

        Ok((scores_list, bboxes_list, kpss_list))
    }

    /// Detect faces in the image
    /// # Arguments:
    /// - image: &RgbImage
    /// - max_num: usize
    /// - metric: &str
    /// # Returns:
    /// - Result<(Array2<f32>, Option<Array3<f32>), Box<dyn Error>>
    pub fn detect(
        &mut self,
        image: &RgbImage,
        max_num: usize,
        metric: &str,
        center_cache: &mut HashMap<(i32, i32, i32), Array2<f32>>,
    ) -> Result<(Array2<f32>, Option<Array3<f32>>), Box<dyn Error>> {
        let orig_width = image.width() as f32;
        let orig_height = image.height() as f32;

        let (input_width, input_height) = (self.input_size.0 as u32, self.input_size.1 as u32);

        let im_ratio = orig_height / orig_width;
        let model_ratio = input_height as f32 / input_width as f32;

        let (new_width, new_height) = if im_ratio > model_ratio {
            let new_height = input_height;
            let new_width = (new_height as f32 / im_ratio).round() as u32;
            (new_width, new_height)
        } else {
            let new_width = input_width;
            let new_height = (new_width as f32 * im_ratio).round() as u32;
            (new_width, new_height)
        };

        let det_scale = new_height as f32 / orig_height;
        println!("Det scale: {}", det_scale);
        let resized_image = image::imageops::resize(
            image,
            new_width,
            new_height,
            image::imageops::FilterType::Triangle,
        );

        // Create a new image with padding (black background)
        let mut det_image = RgbImage::from_pixel(input_width, input_height, image::Rgb([0, 0, 0]));

        // Place the resized image at the top-left corner
        image::imageops::overlay(&mut det_image, &resized_image, 0, 0);
        let input_tensor = self.prepare_input_tensor(&det_image)?;
        let (scores_list, bboxes_list, kpss_list) = self.forward(&input_tensor.into_dyn(), center_cache)?;

        // Concatenate scores and bboxes
        let scores = ScrfdHelpers::concatenate_array2(&scores_list)?;
        println!("Concatenated scores: {:?}", scores);
        let bboxes = ScrfdHelpers::concatenate_array2(&bboxes_list)?;
        let bboxes = &bboxes / det_scale;
        println!("Scaled bboxes: {:?}", bboxes);

        let mut kpss = if self.use_kps {
            let kpss = ScrfdHelpers::concatenate_array3(&kpss_list)?;
            Some(&kpss / det_scale)
        } else {
            None
        };

        let scores_ravel = scores.iter().collect::<Vec<_>>();
        let mut order = (0..scores_ravel.len()).collect::<Vec<usize>>();
        order.sort_unstable_by(|&i, &j| scores_ravel[j].partial_cmp(&scores_ravel[i]).unwrap());
        println!("Order: {:?}", order);

        // Prepare pre_detections
        let mut pre_det = ndarray::concatenate(Axis(1), &[bboxes.view(), scores.view()])?;
        pre_det = pre_det.select(Axis(0), &order);

        let keep = ScrfdHelpers::nms(&pre_det, self.iou_thres);
        let det = pre_det.select(Axis(0), &keep);

        if self.use_kps {
            if let Some(ref mut kpss_array) = kpss {
                *kpss_array = kpss_array.select(Axis(0), &order);
                *kpss_array = kpss_array.select(Axis(0), &keep);
            }
        }

        let det = if max_num > 0 && max_num < det.shape()[0] {
            let area = (&det.slice(s![.., 2]) - &det.slice(s![.., 0]))
                * (&det.slice(s![.., 3]) - &det.slice(s![.., 1]));
            let image_center = (input_width as f32 / 2.0, input_height as f32 / 2.0);
            let offsets = ndarray::stack![
                Axis(0),
                (&det.slice(s![.., 0]) + &det.slice(s![.., 2])) / 2.0 - image_center.1 as f32,
                (&det.slice(s![.., 1]) + &det.slice(s![.., 3])) / 2.0 - image_center.0 as f32,
            ];
            let offset_dist_squared = offsets.mapv(|x| x * x).sum_axis(Axis(0));
            let values = if metric == "max" {
                area.to_owned()
            } else {
                &area - &(offset_dist_squared * 2.0)
            };
            let mut bindex = (0..values.len()).collect::<Vec<usize>>();
            bindex.sort_unstable_by(|&i, &j| values[j].partial_cmp(&values[i]).unwrap());
            bindex.truncate(max_num);
            let det = det.select(Axis(0), &bindex);
            if self.use_kps {
                if let Some(ref mut kpss_array) = kpss {
                    *kpss_array = kpss_array.select(Axis(0), &bindex);
                }
            }
            det
        } else {
            det
        };

        let bounding_boxes =
            ScrfdHelpers::absolute_to_relative_bboxes(&det, orig_width as u32, orig_height as u32);
        let keypoints = if let Some(kpss) = kpss {
            Some(ScrfdHelpers::absolute_to_relative_keypoints(
                &kpss,
                orig_width as u32,
                orig_height as u32,
            ))
        } else {
            None
        };
        Ok((bounding_boxes, keypoints))
    }
}
