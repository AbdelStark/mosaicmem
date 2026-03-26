use mosaicmem::backend::{
    BackendError, BackendMode, BackendRequest, BackendResponse, TensorDType, TensorPayload,
    validate_backend_configuration,
};
use mosaicmem::pipeline::config::PipelineConfig;
use std::path::PathBuf;

#[test]
fn test_backend_bridge_payload_roundtrip() {
    let payload = TensorPayload::from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
    let request = BackendRequest::VaeEncode {
        frames: payload.clone(),
    };
    let response = BackendResponse::Tensor(payload.clone());

    let request_json = serde_json::to_string(&request).unwrap();
    let response_json = serde_json::to_string(&response).unwrap();

    let decoded_request: BackendRequest = serde_json::from_str(&request_json).unwrap();
    let decoded_response: BackendResponse = serde_json::from_str(&response_json).unwrap();

    assert_eq!(decoded_request, request);
    assert_eq!(decoded_response, response);
    assert_eq!(payload.dtype, TensorDType::F32);
    assert_eq!(payload.shape, vec![1, 2, 2]);
}

#[test]
fn test_backend_mode_real_config_roundtrip_and_missing_checkpoint() {
    let config = PipelineConfig {
        backend_mode: BackendMode::Real,
        checkpoint_path: Some(PathBuf::from("/tmp/nonexistent-checkpoint")),
        ..Default::default()
    };

    let json = serde_json::to_string(&config).unwrap();
    let decoded: PipelineConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(decoded.backend_mode, BackendMode::Real);
    assert_eq!(
        decoded.checkpoint_path,
        Some(PathBuf::from("/tmp/nonexistent-checkpoint"))
    );

    let missing = validate_backend_configuration(BackendMode::Real, None);
    assert!(matches!(
        missing,
        Err(BackendError::CheckpointNotFound { .. })
    ));
}
