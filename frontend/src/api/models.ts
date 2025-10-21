const BASE_URL = 'http://127.0.0.1:8000';

export interface AvailableModel {
  id: string;
  run_id: string;
  checkpoint_name: string;
  checkpoint_kind: string;
  path: string;
  size_mb: number;
  model_type: string;
  modified_at: number;
}

export async function getAvailableModels(): Promise<AvailableModel[]> {
  const response = await fetch(`${BASE_URL}/api/models/available`, {
    headers: {
      'Accept': 'application/json',
      'Accept-Charset': 'utf-8',
    },
  });

  if (!response.ok) {
    throw new Error(`Failed to get available models: ${response.statusText}`);
  }

  return response.json();
}
