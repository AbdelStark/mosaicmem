//! Typed tensor views for paper-faithful layout handling.

use ndarray::{ArrayD, Axis, IxDyn};
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensorLayout {
    BCTHW,
    BCHW,
    CTHW,
    CHW,
    THW,
    HW,
    Flat(Vec<usize>),
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum TensorError {
    #[error("tensor rank mismatch for {layout:?}: expected {expected}, got {actual}")]
    RankMismatch {
        layout: TensorLayout,
        expected: usize,
        actual: usize,
    },
    #[error("tensor shape mismatch for flat layout: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    #[error("tensor index {index} is out of bounds for axis length {len}")]
    IndexOutOfBounds { index: usize, len: usize },
    #[error("invalid tensor shape data: {0}")]
    InvalidShapeData(String),
    #[error("layout {0:?} does not support frame extraction")]
    UnsupportedFrameExtraction(TensorLayout),
}

#[derive(Debug, Clone)]
pub struct TensorView {
    data: ArrayD<f32>,
    layout: TensorLayout,
}

impl TensorView {
    pub fn new(data: ArrayD<f32>, layout: TensorLayout) -> Result<Self, TensorError> {
        Self::validate_layout(data.shape(), &layout)?;
        Ok(Self { data, layout })
    }

    pub fn from_shape_vec(
        shape: &[usize],
        values: Vec<f32>,
        layout: TensorLayout,
    ) -> Result<Self, TensorError> {
        let data = ArrayD::from_shape_vec(IxDyn(shape), values)
            .map_err(|err| TensorError::InvalidShapeData(err.to_string()))?;
        Self::new(data, layout)
    }

    pub fn layout(&self) -> &TensorLayout {
        &self.layout
    }

    pub fn data(&self) -> &ArrayD<f32> {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut ArrayD<f32> {
        &mut self.data
    }

    pub fn shape(&self) -> &[usize] {
        self.data.shape()
    }

    pub fn spatial_shape(&self) -> (usize, usize) {
        let shape = self.data.shape();
        if shape.len() < 2 {
            return (0, 0);
        }
        (shape[shape.len() - 2], shape[shape.len() - 1])
    }

    pub fn frame(&self, t: usize) -> Result<Self, TensorError> {
        match &self.layout {
            TensorLayout::BCTHW => {
                let shape = self.shape();
                if t >= shape[2] {
                    return Err(TensorError::IndexOutOfBounds {
                        index: t,
                        len: shape[2],
                    });
                }
                let batch0 = self.data.index_axis(Axis(0), 0).to_owned();
                let frame = batch0.index_axis(Axis(1), t).to_owned().into_dyn();
                Self::new(frame, TensorLayout::CHW)
            }
            TensorLayout::CTHW => {
                let shape = self.shape();
                if t >= shape[1] {
                    return Err(TensorError::IndexOutOfBounds {
                        index: t,
                        len: shape[1],
                    });
                }
                let frame = self.data.index_axis(Axis(1), t).to_owned().into_dyn();
                Self::new(frame, TensorLayout::CHW)
            }
            layout => Err(TensorError::UnsupportedFrameExtraction(layout.clone())),
        }
    }

    pub fn latent_slice(&self, t: usize) -> Result<Self, TensorError> {
        self.frame(t)
    }

    fn validate_layout(shape: &[usize], layout: &TensorLayout) -> Result<(), TensorError> {
        let actual = shape.len();
        let expected = match layout {
            TensorLayout::BCTHW => Some(5),
            TensorLayout::BCHW => Some(4),
            TensorLayout::CTHW => Some(4),
            TensorLayout::CHW => Some(3),
            TensorLayout::THW => Some(3),
            TensorLayout::HW => Some(2),
            TensorLayout::Flat(expected) => {
                if expected.as_slice() != shape {
                    return Err(TensorError::ShapeMismatch {
                        expected: expected.clone(),
                        actual: shape.to_vec(),
                    });
                }
                Some(expected.len())
            }
        };

        if let Some(expected) = expected
            && expected != actual
        {
            return Err(TensorError::RankMismatch {
                layout: layout.clone(),
                expected,
                actual,
            });
        }

        Ok(())
    }
}
