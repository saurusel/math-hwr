// Training types based on Train-and-Models-Spec

export type ModelType = 'M1' | 'M2';
export type JobStatus = 'QUEUED' | 'RUNNING' | 'STOPPING' | 'FINISHED' | 'FAILED' | 'STOPPED';

export interface TrainingConfig {
  model_type: ModelType;
  run_name: string;
  dataset: {
    name: string;
    train_manifest: string;
    val_manifest: string;
  };
  hyper: {
    batch_size: number;
    epochs: number;
    img_h: number;
    img_w_max: number;
    lr: number;
    optimizer: string;
    scheduler: string;
    seed: number;
    advanced?: {
      m1?: { weight_decay?: number };
      m2?: {
        dropout?: number;
        min_seg_area?: number;
        max_seg_area?: number;
        target_char_size?: number;
      };
    };
  };
  augment: {
    invert: boolean;
    random_pad: boolean;
  };
}

export interface TrainingJob {
  job_id: string;
  status: JobStatus;
  model_type: ModelType;
  run_name: string;
  started_at?: string;
  finished_at?: string;
  progress?: number;
  current_epoch?: number;
  best_metric?: Record<string, number>;
  last_checkpoint?: string;
}

export interface MetricEvent {
  epoch: number;
  train_loss: number;
  val_loss: number;
  cer?: number;
  wer?: number;
  exact?: number;
  valid?: number;
  lr: number;
  epoch_time_sec?: number;
  samples_processed?: number;
  batches_processed?: number;
  grad_norm?: number;
}

export interface LogEvent {
  ts: string;
  line: string;
}

export interface SamplePrediction {
  id: string;
  target: string;
  pred: string;
  image_b64?: string;  // Base64 encoded image
  png_path?: string;   // File path (for reference)
  ok?: boolean;        // Match indicator
}

export interface SamplePredEvent {
  epoch: number;
  items: SamplePrediction[];
}

export interface CheckpointEvent {
  path: string;
  kind: 'best' | 'last';
  epoch?: number;
  metric?: Record<string, number>;
}

export interface StatusEvent {
  status: JobStatus;
}

export interface Experiment {
  run_id: string;
  run_name: string;
  model_type: ModelType;
  status: JobStatus;
  started_at: string;
  finished_at?: string;
  best_metrics?: Record<string, number>;
  config?: TrainingConfig;
}

export interface Checkpoint {
  ckpt_id: string;
  path: string;
  size: number;
  epoch: number;
  kind: 'best' | 'last' | 'epoch';
  metrics_at_save?: Record<string, number>;
}
