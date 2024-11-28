use ndarray::{Array2, Array3, Axis};
use std::cmp::Ordering;
use std::error::Error;

pub struct ScrfdHelpers;

impl ScrfdHelpers {
    /// Decode distance prediction to bounding box.
    ///
    /// # Arguments
    ///
    /// * `points` - Shape (n, 2), [x, y].
    /// * `distance` - Distance from the given point to 4 boundaries (left, top, right, bottom).
    /// * `max_shape` - Shape of the image.
    ///
    /// # Returns
    ///
    /// * `Array2<f32>` - Decoded bounding boxes with shape (n, 4).
    pub fn distance2bbox(
        points: &Array2<f32>,
        distance: &Array2<f32>,
        max_shape: Option<(usize, usize)>,
    ) -> Array2<f32> {
        let x1 = &points.column(0) - &distance.column(0);
        let y1 = &points.column(1) - &distance.column(1);
        let x2 = &points.column(0) + &distance.column(2);
        let y2 = &points.column(1) + &distance.column(3);

        // Optionally clamp the values if max_shape is provided
        let (x1, y1, x2, y2) = if let Some((height, width)) = max_shape {
            let width = width as f32;
            let height = height as f32;
            let x1 = x1.mapv(|x| x.max(0.0).min(width));
            let y1 = y1.mapv(|y| y.max(0.0).min(height));
            let x2 = x2.mapv(|x| x.max(0.0).min(width));
            let y2 = y2.mapv(|y| y.max(0.0).min(height));
            (x1, y1, x2, y2)
        } else {
            // Do not clamp if max_shape is None
            (x1, y1, x2, y2)
        };

        println!("x1: {:?}", x1);
        println!("y1: {:?}", y1);
        println!("x2: {:?}", x2);
        println!("y2: {:?}", y2);

        let concatenated =
            ndarray::stack(Axis(1), &[x1.view(), y1.view(), x2.view(), y2.view()]).unwrap();
        println!("Shape: {:?}", concatenated.shape());
        concatenated
    }

    /// Decode distance prediction to keypoints.
    ///
    /// # Arguments
    ///
    /// * `points` - Shape (n, 2), [x, y].
    /// * `distance` - Distance from the given point to keypoints.
    /// * `max_shape` - Shape of the image.
    ///
    /// # Returns
    ///
    /// * `Array2<f32>` - Decoded keypoints with shape (n, 10).
    pub fn distance2kps(
        points: &Array2<f32>,
        distance: &Array2<f32>,
        max_shape: Option<(usize, usize)>,
    ) -> Array2<f32> {
        let num_keypoints = distance.shape()[1] / 2;
        let mut preds = Vec::with_capacity(2 * num_keypoints);

        for i in 0..num_keypoints {
            let px = &points.column(0) + &distance.column(2 * i);
            let py = &points.column(1) + &distance.column(2 * i + 1);
            let (px, py) = if let Some((height, width)) = max_shape {
                let width = width as f32;
                let height = height as f32;
                let px = px.mapv(|x| x.max(0.0).min(width));
                let py = py.mapv(|y| y.max(0.0).min(height));
                (px, py)
            } else {
                (px, py)
            };
            preds.push(px.insert_axis(Axis(1)));
            preds.push(py.insert_axis(Axis(1)));
        }

        // Concatenate along Axis(1) to get an array of shape (n, 2k)
        ndarray::concatenate(Axis(1), &preds.iter().map(|a| a.view()).collect::<Vec<_>>()).unwrap()
    }
    /// Non-Maximum Suppression (NMS) function
    /// # Arguments
    /// * `dets` - Detection results with shape (n, 5), [x1, y1, x2, y2, score].
    /// * `iou_thres` - IoU threshold to suppress overlapping boxes.
    ///
    /// # Returns
    /// * `Vec<usize>` - Indices of the boxes to keep.
    pub fn nms(dets: &Array2<f32>, iou_thres: f32) -> Vec<usize> {
        let x1 = dets.column(0);
        let y1 = dets.column(1);
        let x2 = dets.column(2);
        let y2 = dets.column(3);
        let scores = dets.column(4);

        let areas = (&x2 - &x1 + 1.0) * (&y2 - &y1 + 1.0);
        let mut order: Vec<usize> = (0..scores.len()).collect();
        order.sort_unstable_by(|&i, &j| {
            scores[j].partial_cmp(&scores[i]).unwrap_or(Ordering::Equal)
        });

        let mut keep = Vec::new();
        while !order.is_empty() {
            let i = order[0];
            keep.push(i);

            if order.len() == 1 {
                break;
            }

            let order_rest = &order[1..];

            // Extract scalar values
            let x1_i = x1[i];
            let y1_i = y1[i];
            let x2_i = x2[i];
            let y2_i = y2[i];
            let area_i = areas[i];

            // Select the rest of the array
            let x1_order = x1.select(Axis(0), order_rest);
            let y1_order = y1.select(Axis(0), order_rest);
            let x2_order = x2.select(Axis(0), order_rest);
            let y2_order = y2.select(Axis(0), order_rest);
            let areas_order = areas.select(Axis(0), order_rest);

            // Compute the coordinates of the intersection
            let xx1 = x1_order.mapv(|x| x1_i.max(x));
            let yy1 = y1_order.mapv(|y| y1_i.max(y));
            let xx2 = x2_order.mapv(|x| x2_i.min(x));
            let yy2 = y2_order.mapv(|y| y2_i.min(y));

            // Compute the width and height of the intersection
            let w = (&xx2 - &xx1 + 1.0).mapv(|x| x.max(0.0));
            let h = (&yy2 - &yy1 + 1.0).mapv(|y| y.max(0.0));
            let inter = &w * &h;
            let ovr = &inter / (area_i + &areas_order - &inter);

            // Get indices where IoU <= threshold
            let inds: Vec<usize> = ovr
                .iter()
                .enumerate()
                .filter(|&(_, &ov)| ov <= iou_thres)
                .map(|(idx, _)| idx)
                .collect();

            // Update order
            let mut new_order = Vec::with_capacity(inds.len());
            for &idx in &inds {
                new_order.push(order[idx + 1]); // +1 because we skipped order[0]
            }
            order = new_order;
        }
        keep
    }

    /// Generate anchor centers for a feature map.
    /// # Arguments
    /// * `num_anchors` - Number of anchors per location.
    /// * `height` - Height of the feature map.
    /// * `width` - Width of the feature map.
    /// * `stride` - Stride of the feature map.
    ///
    /// # Returns
    /// * `Array2<f32>` - Anchor centers with shape (height * width * num_anchors, 2).
    pub fn generate_anchor_centers(
        num_anchors: usize,
        height: usize,
        width: usize,
        stride: f32,
    ) -> Array2<f32> {
        // Create anchor centers using [x, y] ordering
        let mut anchor_centers = Array2::zeros((height * width, 2));

        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                anchor_centers[[idx, 0]] = x as f32; // Assign x first
                anchor_centers[[idx, 1]] = y as f32; // Assign y second
            }
        }

        // Multiply by stride
        let anchor_centers = anchor_centers.mapv(|x| x * stride);

        // Handle multiple anchors if needed
        let anchor_centers = if num_anchors > 1 {
            let mut repeated_anchors = Array2::zeros((height * width * num_anchors, 2));

            // Repeat each point num_anchors times
            for (i, row) in anchor_centers.rows().into_iter().enumerate() {
                for j in 0..num_anchors {
                    repeated_anchors
                        .slice_mut(ndarray::s![i * num_anchors + j, ..])
                        .assign(&row);
                }
            }

            repeated_anchors
        } else {
            anchor_centers
        };

        anchor_centers
    }

    /// Helper function to concatenate a vector of Array2<f32>
    /// # Arguments
    /// * `arrays` - Vector of Array2<f32> to concatenate.
    /// # Returns
    /// * `Result<Array2<f32>, Box<dyn Error>>` - Concatenated array.
    pub fn concatenate_array2(arrays: &[Array2<f32>]) -> Result<Array2<f32>, Box<dyn Error>> {
        if arrays.is_empty() {
            return Ok(Array2::<f32>::zeros((0, arrays[0].shape()[1])));
        }
        Ok(ndarray::concatenate(
            Axis(0),
            &arrays.iter().map(|a| a.view()).collect::<Vec<_>>(),
        )?)
    }

    /// Helper function to concatenate a vector of Array3<f32>
    /// # Arguments
    /// * `arrays` - Vector of Array3<f32> to concatenate.
    /// # Returns
    /// * `Result<Array3<f32>, Box<dyn Error>>` - Concatenated array.
    pub fn concatenate_array3(arrays: &[Array3<f32>]) -> Result<Array3<f32>, Box<dyn Error>> {
        if arrays.is_empty() {
            return Ok(Array3::<f32>::zeros((
                0,
                arrays[0].shape()[1],
                arrays[0].shape()[2],
            )));
        }
        Ok(ndarray::concatenate(
            Axis(0),
            &arrays.iter().map(|a| a.view()).collect::<Vec<_>>(),
        )?)
    }

    /// Converts absolute bounding boxes to relative coordinates.
    ///
    /// # Arguments
    ///
    /// * `bboxes` - A reference to a 2D array containing bounding boxes in `[x1, y1, x2, y2]` format.
    /// * `img_width` - The width of the image in pixels.
    /// * `img_height` - The height of the image in pixels.
    ///
    /// # Returns
    ///
    /// A 2D array containing bounding boxes in `[Left, Top, Width, Height]` format with values normalized between 0 and 1.
    pub fn absolute_to_relative_bboxes(
        bboxes: &Array2<f32>,
        img_width: u32,
        img_height: u32,
    ) -> Array2<f32> {
        // Convert image dimensions to f32 for division
        let img_width_f = img_width as f32;
        let img_height_f = img_height as f32;

        // Initialize a new Array2 for relative bounding boxes
        let mut relative_bboxes = Array2::<f32>::zeros((bboxes.nrows(), 4));

        for (i, bbox) in bboxes.axis_iter(Axis(0)).enumerate() {
            let x1 = bbox[0];
            let y1 = bbox[1];
            let x2 = bbox[2];
            let y2 = bbox[3];

            // Calculate relative coordinates
            let left = x1 / img_width_f;
            let top = y1 / img_height_f;
            let width = (x2 - x1) / img_width_f;
            let height = (y2 - y1) / img_height_f;

            // Assign to the new array
            relative_bboxes[[i, 0]] = left;
            relative_bboxes[[i, 1]] = top;
            relative_bboxes[[i, 2]] = width;
            relative_bboxes[[i, 3]] = height;
        }

        relative_bboxes
    }

    /// Converts absolute keypoints to relative coordinates.
    ///
    /// # Arguments
    ///
    /// * `keypoints` - A reference to a 3D array containing keypoints in `[x, y]` format.
    ///                 Shape: `(num_detections, num_keypoints, 2)`
    /// * `img_width` - The width of the image in pixels.
    /// * `img_height` - The height of the image in pixels.
    ///
    /// # Returns
    ///
    /// A 3D array containing keypoints in `[x_rel, y_rel]` format with values normalized between 0 and 1.
    pub fn absolute_to_relative_keypoints(
        keypoints: &Array3<f32>,
        img_width: u32,
        img_height: u32,
    ) -> Array3<f32> {
        let img_width_f = img_width as f32;
        let img_height_f = img_height as f32;

        // Initialize a new Array3 for relative keypoints
        let mut relative_keypoints = Array3::<f32>::zeros(keypoints.dim());

        for (i, kp_set) in keypoints.axis_iter(Axis(0)).enumerate() {
            for (j, kp) in kp_set.axis_iter(Axis(0)).enumerate() {
                let x_rel = kp[0] / img_width_f;
                let y_rel = kp[1] / img_height_f;

                // Clamp values between 0 and 1 to handle edge cases
                relative_keypoints[[i, j, 0]] = x_rel.clamp(0.0, 1.0);
                relative_keypoints[[i, j, 1]] = y_rel.clamp(0.0, 1.0);
            }
        }

        relative_keypoints
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_nms() {
        let dets = array![
            [10.0, 10.0, 20.0, 20.0, 0.9],
            [12.0, 12.0, 22.0, 22.0, 0.8],
            [15.0, 15.0, 25.0, 25.0, 0.7],
            [30.0, 30.0, 40.0, 40.0, 0.6],
        ];
        let iou_thres = 0.5;
        let keep = ScrfdHelpers::nms(&dets, iou_thres);
        assert_eq!(keep, vec![0, 2, 3]); // Updated expected result
    }

    #[test]
    fn test_distance2bbox() {
        let points = array![[10.0, 10.0], [20.0, 20.0]];
        let distance = array![[1.0, 1.0, 2.0, 2.0], [3.0, 3.0, 4.0, 4.0]];
        let expected = array![[9.0, 9.0, 12.0, 12.0], [17.0, 17.0, 24.0, 24.0]];
        let bbox = ScrfdHelpers::distance2bbox(&points, &distance, Some((30, 30)));
        assert_eq!(bbox, expected);
    }

    #[test]
    fn test_distance2kps() {
        let points = array![[10.0, 10.0], [20.0, 20.0]];
        let distance = array![
            [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0],
            [6.0, 6.0, 7.0, 7.0, 8.0, 8.0, 9.0, 9.0, 10.0, 10.0]
        ];
        let expected = array![
            [11.0, 11.0, 12.0, 12.0, 13.0, 13.0, 14.0, 14.0, 15.0, 15.0],
            [26.0, 26.0, 27.0, 27.0, 28.0, 28.0, 29.0, 29.0, 30.0, 30.0]
        ];
        let kps = ScrfdHelpers::distance2kps(&points, &distance, Some((30, 30)));
        assert_eq!(kps, expected);
    }

    #[test]
    fn test_distance2bbox_stride_1() {
        let anchor_centers = array![[100.0, 100.0]];
        let bbox_preds = array![[0.1, 0.1, 0.2, 0.2]];
        let expected_bboxes = array![[99.9, 99.9, 100.2, 100.2]];

        let bboxes = ScrfdHelpers::distance2bbox(&anchor_centers, &bbox_preds, None);
        assert_eq!(bboxes, expected_bboxes);
    }

    #[test]
    fn test_generate_anchor_centers_multiple_anchors() {
        let height = 2;
        let width = 2;
        let stride = 16.0;
        let num_anchors = 2;

        let anchor_centers =
            ScrfdHelpers::generate_anchor_centers(num_anchors, height, width, stride);

        let expected = array![
            [0.0, 0.0],
            [0.0, 0.0],
            [16.0, 0.0],
            [16.0, 0.0],
            [0.0, 16.0],
            [0.0, 16.0],
            [16.0, 16.0],
            [16.0, 16.0]
        ];

        println!("Anchor centers: {:?}", anchor_centers);
        println!("Expected: {:?}", expected);

        assert_eq!(anchor_centers, expected);
    }
}
