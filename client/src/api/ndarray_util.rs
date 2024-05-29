/// ndarray helper functions. These include image transormations and conversions.
use anyhow::Result;
use image::{Rgba, RgbaImage};
use ndarray::{Array3, Axis};
use std::io::Cursor;

use super::interfaces::CameraAdjustment;

/// Helper function which converts a given jpeg image byte array into a
/// 3D array, which includes RGBA values for each pixel.
///
/// Args:
/// * image_data: JPEG image u8 array.
///
/// Returns:
/// * A converted 3DArray instance.
pub fn image_bytes_to_ndarray(image_data: &[u8]) -> Result<(Array3<u8>, RgbaImage)> {
  // Load the image and then convert to a 3D Array.
  let image = image::load_from_memory(&image_data)?.to_rgba8();
  let (width, height) = image.dimensions();

  // Convert to ndarray to manually modify the image.
  // 3rd dimension are the rgba values of each pixel.
  let mut image_ndarray: Array3<u8> = Array3::zeros((height as usize, width as usize, 4));
  for y in 0..height {
    for x in 0..width {
      let pixel: &Rgba<u8> = image.get_pixel(x, y);

      // Set all rgba values for each pixel.
      image_ndarray[(y as usize, x as usize, 0 as usize)] = pixel[0];
      image_ndarray[(y as usize, x as usize, 1 as usize)] = pixel[1];
      image_ndarray[(y as usize, x as usize, 2 as usize)] = pixel[2];
      image_ndarray[(y as usize, x as usize, 3 as usize)] = pixel[3];
    }
  }

  Ok((image_ndarray, image))
}

/// Helper fucntion which converts a given ndarray back back into an
/// encoded jpeg image.
///
/// Args:
/// * image_ndarray: 3DArray instance to be converted into a u8 array.
///
/// Returns:
/// * JPEG image u8 array.
pub fn ndarray_to_image_bytes(image_ndarray: Array3<u8>) -> Result<Vec<u8>> {
  let (height, width) = (image_ndarray.shape()[0], image_ndarray.shape()[1]);

  // Consume the ndarray into an image buffer of which we can then encode
  // into a jpeg image buffer.
  let mut image_buff = RgbaImage::new(width as u32, height as u32);
  for (x, y, pixel) in image_buff.enumerate_pixels_mut() {
    // Set all rgba values back into each pixel.
    pixel[0] = image_ndarray[(y as usize, x as usize, 0 as usize)];
    pixel[1] = image_ndarray[(y as usize, x as usize, 1 as usize)];
    pixel[2] = image_ndarray[(y as usize, x as usize, 2 as usize)];
    pixel[3] = image_ndarray[(y as usize, x as usize, 3 as usize)];
  }

  let mut mem_buff = Cursor::new(vec![]);
  image_buff.write_to(&mut mem_buff, image::ImageFormat::Jpeg)?;
  Ok(mem_buff.get_ref().to_vec())
}

/// Helper function which crops a given ndarray image using the adjustment struct.
///
/// Args:
/// * ndarray_image: 3DArray instance to be cropped.
/// * image: RgbImage instance used for image metadata.
/// * adjustment_config: Reference to a CameraAdjustment struct for cropping info.
///
/// Returns:
/// * Cropped 3DAarray instance.
pub fn ndarray_crop_image(
  mut ndarray_image: Array3<u8>,
  image: RgbaImage,
  adjustment_config: &CameraAdjustment,
) -> Array3<u8> {
  let (width, height) = image.dimensions();

  // Grab/calculate adjustments.
  let height_adjustment =
    math::round::floor((height as f64) * adjustment_config.crop_frame_height, 3) as u32;
  let width_adjustment =
    math::round::floor((width as f64) * adjustment_config.crop_frame_width, 3) as u32;
  let (crop_x, crop_y) = (
    adjustment_config.crop_frame_x as usize,
    adjustment_config.crop_frame_y as usize,
  );

  // Finally, apply crop changes.
  let new_height = (height - height_adjustment) as usize;
  let new_width = (width - width_adjustment) as usize;

  let y0 = crop_y as usize;
  let y1 = (crop_y + new_height) as usize;
  let x0 = crop_x as usize;
  let x1 = (crop_x + new_width) as usize;

  // Slice the width and height, retaining the channel indicies.
  return ndarray_image
    .slice_mut(ndarray::s![y0..y1, x0..x1, ..])
    .to_owned();
}

/// Helper function which rotates a given ndarray image counter-clockwise.
///
/// Args:
/// * ndarray_image: 3DArray instance to rotate.
/// * degrees: Degrees to rotate the image by.
///
/// Returns:
/// * Rotated 3DAarray instance.
pub fn ndarray_rotate_image(ndarray_image: Array3<u8>, degrees: f64) -> Array3<u8> {
  // Convert to radians.
  let rot_rad = degrees.to_radians();

  // Grab original image dimensions.
  let img_width = ndarray_image.len_of(Axis(1));
  let img_height = ndarray_image.len_of(Axis(0));

  // Calculate new dimensions after rotation.
  let rot_width =
    (img_width as f64 * rot_rad.cos() + img_height as f64 * rot_rad.sin()).abs() as usize;
  let rot_height =
    (img_height as f64 * rot_rad.cos() + img_width as f64 * rot_rad.sin()).abs() as usize;

  // Setup for rotation op.
  let mut rotated_img: Array3<u8> = Array3::default((rot_height, rot_width, 4));

  // Calculate the center of the original image.
  let center_img_x: f64 = (img_width as f64) / 2.0;
  let center_img_y: f64 = (img_height as f64) / 2.0;

  // Calculate the center of the rotated image.
  let center_rot_img_x: f64 = (rot_width as f64) / 2.0;
  let center_rot_img_y: f64 = (rot_height as f64) / 2.0;

  // Apply rotation by mapping each pixel from the original image.
  for ((y, x, rgba_idx), value) in rotated_img.indexed_iter_mut() {
    let rot_x = x as f64 - center_rot_img_x;
    let rot_y = center_rot_img_y - y as f64;

    let original_x =
      (rot_x * rot_rad.cos() - rot_y * rot_rad.sin() + center_img_x).round() as isize;
    let original_y =
      (rot_x * rot_rad.sin() - rot_y * rot_rad.cos() + center_img_y).round() as isize;

    if original_x >= 0
      && original_x < img_width as isize
      && original_y >= 0
      && original_y < img_height as isize
    {
      *value = ndarray_image[[original_y as usize, original_x as usize, rgba_idx]].clone();
    }
  }

  return rotated_img;
}
