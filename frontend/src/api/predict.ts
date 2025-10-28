import { apiPost } from './client';

export interface PredictRequest {
  run_id?: string;
  checkpoint?: string;
  data: {
    img_h: number;
    img_w_max: number;
    invert: boolean;
    pad_mode: string;
    return_b64_preprocessed?: boolean;
  };
  decode: {
    type: 'greedy' | 'beam';
    beam_width?: number;
  };
  image: {
    b64: string;
  };
}

export interface SegmentInfo {
  id: number;
  bbox: {
    x: number;
    y: number;
    w: number;
    h: number;
  };
  token: string;
  token_id: number;
}

export interface PredictResponse {
  device: string;
  ckpt: string;
  img_h?: number;
  img_w_max?: number;
  decoder?: string;
  tokens: string[];
  text: string;
  preprocessed_b64?: string;
  model_type?: string;
  num_segments?: number;
  segments?: SegmentInfo[];  // NEW: Segmentation info for M2
  image_size?: { width: number; height: number };  // NEW: Original image dimensions
  error?: string;  // NEW: Error message
  debug_binary?: string;  // NEW: Debug binarized image
}

export async function predictM1(
  imageDataURL: string,
  checkpointInfo?: { run_id: string; checkpoint_name: string }
): Promise<PredictResponse> {
  const request: PredictRequest = {
    run_id: checkpointInfo?.run_id,
    checkpoint: checkpointInfo?.checkpoint_name,
    data: {
      img_h: 64,
      img_w_max: 256,  // Changed from 512 to 256 (4x uniform scaling)
      invert: true,
      pad_mode: 'center',  // Changed to center for better alignment
      return_b64_preprocessed: true,
    },
    decode: {
      type: 'greedy',
    },
    image: {
      b64: imageDataURL,
    },
  };

  return apiPost<PredictResponse>('/api/predict2/run', request);
}

export async function predictM2(
  imageDataURL: string,
  checkpointInfo?: { run_id: string; checkpoint_name: string }
): Promise<PredictResponse> {
  const request: PredictRequest = {
    run_id: checkpointInfo?.run_id,
    checkpoint: checkpointInfo?.checkpoint_name,
    data: {
      img_h: 64,
      img_w_max: 256,  // Changed from 512 to 256 (4x uniform scaling)
      invert: true,
      pad_mode: 'center',  // Changed to center for better alignment
      return_b64_preprocessed: false,  // Don't need preprocessed for M2
    },
    decode: {
      type: 'greedy',
    },
    image: {
      b64: imageDataURL,
    },
  };

  return apiPost<PredictResponse>('/api/predict2/run', request);  // Use unified endpoint
}

export async function predictM3(
  imageDataURL: string,
  checkpointInfo?: { run_id: string; checkpoint_name: string }
): Promise<PredictResponse> {
  const request = {
    run_id: checkpointInfo?.run_id,
    checkpoint: checkpointInfo?.checkpoint_name,
    image: {
      b64: imageDataURL,
    },
    data: {
      img_h: 64,
      img_w_max: 512,
    },
  };

  return apiPost<PredictResponse>('/api/predict2/vit/run', request);
}
