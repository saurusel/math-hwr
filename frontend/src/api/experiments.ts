import { apiPost } from './client';
import type { TrainingConfig, Checkpoint } from '../types/training';

const BASE_URL = 'http://127.0.0.1:8000';

export async function duplicateRun(runId: string): Promise<TrainingConfig> {
  const response = await fetch(`${BASE_URL}/api/experiments/${runId}/config`, {
    headers: {
      'Accept': 'application/json',
      'Accept-Charset': 'utf-8',
    },
  });

  if (!response.ok) {
    throw new Error(`Failed to get run config: ${response.statusText}`);
  }

  const config: TrainingConfig = await response.json();

  // Modify run_name to indicate duplication
  config.run_name = `${config.run_name}_copy_${Date.now()}`;

  return config;
}

export async function deleteCheckpoint(ckptId: string): Promise<void> {
  const response = await fetch(`${BASE_URL}/api/models/${ckptId}`, {
    method: 'DELETE',
    headers: {
      'Accept': 'application/json',
      'Accept-Charset': 'utf-8',
    },
  });

  if (!response.ok) {
    throw new Error(`Failed to delete checkpoint: ${response.statusText}`);
  }
}

export async function addRunTag(runId: string, tag: string): Promise<void> {
  await apiPost(`/api/experiments/${runId}/tags`, { tag });
}

export async function removeRunTag(runId: string, tag: string): Promise<void> {
  const response = await fetch(`${BASE_URL}/api/experiments/${runId}/tags/${tag}`, {
    method: 'DELETE',
  });

  if (!response.ok) {
    throw new Error(`Failed to remove tag: ${response.statusText}`);
  }
}

export async function updateRunNotes(runId: string, notes: string): Promise<void> {
  await apiPost(`/api/experiments/${runId}/notes`, { notes });
}

export async function getArtifacts(runId: string, epoch?: number): Promise<string[]> {
  const queryParam = epoch !== undefined ? `?epoch=${epoch}` : '';
  const response = await fetch(`${BASE_URL}/api/experiments/${runId}/artifacts${queryParam}`, {
    headers: {
      'Accept': 'application/json',
      'Accept-Charset': 'utf-8',
    },
  });

  if (!response.ok) {
    throw new Error(`Failed to get artifacts: ${response.statusText}`);
  }

  return response.json();
}
