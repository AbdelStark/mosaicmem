use mosaicmem::{TensorError, TensorLayout, TensorView};

#[test]
fn test_tensor_view_construction_and_layout_validation() {
    let bcthw = TensorView::from_shape_vec(
        &[1, 2, 3, 4, 5],
        (0..120).map(|i| i as f32).collect(),
        TensorLayout::BCTHW,
    )
    .unwrap();
    assert_eq!(bcthw.spatial_shape(), (4, 5));

    let chw =
        TensorView::from_shape_vec(&[3, 8, 8], vec![1.0; 3 * 8 * 8], TensorLayout::CHW).unwrap();
    assert_eq!(chw.spatial_shape(), (8, 8));

    let hw = TensorView::from_shape_vec(&[6, 7], vec![0.0; 42], TensorLayout::HW).unwrap();
    assert_eq!(hw.spatial_shape(), (6, 7));

    let err = TensorView::from_shape_vec(&[1, 2, 3], vec![0.0; 6], TensorLayout::BCHW).unwrap_err();
    assert!(matches!(err, TensorError::RankMismatch { .. }));
}

#[test]
fn test_tensor_view_frame_extraction_roundtrip() {
    let tensor = TensorView::from_shape_vec(
        &[1, 2, 3, 2, 2],
        (0..24).map(|i| i as f32).collect(),
        TensorLayout::BCTHW,
    )
    .unwrap();

    let frame = tensor.frame(1).unwrap();
    assert_eq!(frame.layout(), &TensorLayout::CHW);
    assert_eq!(frame.shape(), &[2, 2, 2]);
    assert_eq!(
        frame.data().iter().copied().collect::<Vec<_>>(),
        vec![4.0, 5.0, 6.0, 7.0, 16.0, 17.0, 18.0, 19.0]
    );

    let latent_slice = tensor.latent_slice(1).unwrap();
    assert_eq!(
        latent_slice.data().iter().copied().collect::<Vec<_>>(),
        frame.data().iter().copied().collect::<Vec<_>>()
    );
}
