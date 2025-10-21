const BASE_URL = 'http://127.0.0.1:8000';
const DEFAULT_TIMEOUT = 30000;

export class APIError extends Error {
  constructor(
    message: string,
    public status?: number,
    public response?: string
  ) {
    super(message);
    this.name = 'APIError';
  }
}

async function fetchWithTimeout(
  url: string,
  options: RequestInit,
  timeout: number = DEFAULT_TIMEOUT
): Promise<Response> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout);

  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
    });
    clearTimeout(timeoutId);
    return response;
  } catch (error) {
    clearTimeout(timeoutId);
    if (error instanceof Error && error.name === 'AbortError') {
      throw new APIError('Request timeout');
    }
    throw error;
  }
}

export async function apiPost<T>(
  endpoint: string,
  body: unknown,
  timeout?: number
): Promise<T> {
  try {
    const response = await fetchWithTimeout(
      `${BASE_URL}${endpoint}`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json; charset=utf-8',
          Accept: 'application/json',
          'Accept-Charset': 'utf-8',
        },
        body: JSON.stringify(body),
      },
      timeout
    );

    if (!response.ok) {
      const text = await response.text().catch(() => '');
      throw new APIError(
        `HTTP ${response.status}: ${text || response.statusText}`,
        response.status,
        text
      );
    }

    return await response.json();
  } catch (error) {
    if (error instanceof APIError) {
      throw error;
    }
    if (error instanceof Error) {
      throw new APIError(`Network error: ${error.message}`);
    }
    throw new APIError('Unknown error occurred');
  }
}
