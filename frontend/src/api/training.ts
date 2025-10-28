import { apiPost } from './client';
import type { TrainingConfig, TrainingJob, Experiment, Checkpoint, MetricEvent, LogEvent } from '../types/training';

const BASE_URL = 'http://127.0.0.1:8000';

export async function createTrainingJob(config: TrainingConfig): Promise<{ job_id: string; status: string }> {
  return apiPost<{ job_id: string; status: string }>('/api/train/jobs', config);
}

export async function getJobStatus(jobId: string): Promise<TrainingJob> {
  const response = await fetch(`${BASE_URL}/api/train/jobs/${jobId}`, {
    headers: {
      'Accept': 'application/json',
      'Accept-Charset': 'utf-8',
    },
  });

  if (!response.ok) {
    throw new Error(`Failed to get job status: ${response.statusText}`);
  }

  return response.json();
}

export async function getTrainingHistory(jobId: string): Promise<{
  metrics: MetricEvent[];
  logs: LogEvent[];
  samples: any[];
  checkpoints: any[];
}> {
  const response = await fetch(`${BASE_URL}/api/train/jobs/${jobId}/history`, {
    headers: {
      'Accept': 'application/json',
      'Accept-Charset': 'utf-8',
    },
  });

  if (!response.ok) {
    // If not found, return empty
    if (response.status === 404) {
      return { metrics: [], logs: [], samples: [], checkpoints: [] };
    }
    throw new Error(`Failed to get training history: ${response.statusText}`);
  }

  return response.json();
}

export async function stopJob(jobId: string): Promise<void> {
  await apiPost(`/api/train/jobs/${jobId}/stop`, {});
}

export async function listExperiments(params?: {
  model?: string;
  status?: string;
  q?: string;
  from?: string;
  to?: string;
  tag?: string;
}): Promise<Experiment[]> {
  const queryParams = new URLSearchParams();
  if (params) {
    Object.entries(params).forEach(([key, value]) => {
      if (value) queryParams.append(key, value);
    });
  }

  const url = `${BASE_URL}/api/experiments${queryParams.toString() ? '?' + queryParams.toString() : ''}`;
  const response = await fetch(url, {
    headers: {
      'Accept': 'application/json',
      'Accept-Charset': 'utf-8',
    },
  });

  if (!response.ok) {
    throw new Error(`Failed to list experiments: ${response.statusText}`);
  }

  return response.json();
}

export async function getCheckpoints(runId: string): Promise<Checkpoint[]> {
  const response = await fetch(`${BASE_URL}/api/experiments/${runId}/checkpoints`, {
    headers: {
      'Accept': 'application/json',
      'Accept-Charset': 'utf-8',
    },
  });

  if (!response.ok) {
    throw new Error(`Failed to get checkpoints: ${response.statusText}`);
  }

  return response.json();
}

export async function promoteCheckpoint(ckptId: string, alias: string = 'production'): Promise<void> {
  await apiPost(`/api/models/${ckptId}/promote`, { alias });
}
